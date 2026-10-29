// /src/main.cpp
//
// C++ client for the RVC voice changer. It records a clip from the microphone, sends it to the
// Python inference server (src/server/server.py) and plays the converted voice back.
//
// Threads:
//   - audio callback (miniaudio): mic -> g_rb_input, g_rb_output -> speakers. Never blocks.
//   - worker: drains g_rb_input, accumulates the requested seconds, then runs the clip pipeline:
//       denoise (RNNoise) -> normalize -> resample 48k->16k -> server -> resample model rate->48k
//       -> sample-count correction -> g_rb_output
//   - main: command prompt and all console output.

#include "indicators/indeterminate_progress_bar.hpp"
#include "indicators/cursor_control.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define MINIAUDIO_IMPLEMENTATION // Compile the miniaudio implementation in this translation unit.
#include <miniaudio.h>

#include "Denoiser.h"
#include "PythonBridge.h"
#include "Resampler.h"
#include "RingBuffer.h"

// --- SETTINGS ---
// Everything is mono 48 kHz. miniaudio converts to and from whatever the hardware uses.
constexpr int kSampleRate = 48000;
constexpr int kServerSampleRate = 16000; // HuBERT/ContentVec and pitch detection both analyze 16 kHz audio.
constexpr double kMaxDriftFraction = 0.02; // Larger length mismatches are bugs, not drift, and are left visible.
constexpr size_t kRingBufferSamples = kSampleRate * 30;
constexpr float kMaxRecordSeconds = 25.0f; // A clip must fit in the 30 s output ring buffer.
constexpr float kMaxPitchSemitones = 24.0f;
const char* const kSaveDir = "saved_outputs";

struct AppConfig { // i think i used to have configs in multiple spots which screwed up my debugging?
    float pitch_shift_semitones = 12.0f; // +12 = one octave up.
    float recording_duration = 5.0f;
} g_config;

// --- STATE ---
AudioRingBuffer g_rb_input;  // Audio callback -> worker.
AudioRingBuffer g_rb_output; // Worker or main thread -> audio callback. Only one of them writes at a time.

std::atomic<bool> g_running(true);
std::atomic<bool> g_is_recording(false);     // Set by main to request a clip; cleared by the worker when the clip is done.
std::atomic<bool> g_is_processing(false);    // Worker is waiting for the server.
std::atomic<bool> g_last_ok(false);          // Whether the last request returned audio.
std::atomic<size_t> g_recorded_samples(0);   // Progress of the current recording, for the prompt.
std::atomic<bool> g_denoise(true);           // Toggled with the 'd' command.

// Results of the last clip, published by the worker before it clears g_is_recording.
std::atomic<float> g_last_voice_activity(0.0f); // RNNoise's average speech probability, or -1 when denoising was off.
std::atomic<long> g_last_drift(0);              // Converted length minus recorded length, in 48 kHz samples.
std::atomic<bool> g_last_drift_corrected(false);

// Written only by the worker while g_is_recording is true, read only by main while it is false.
// The atomic flag hand-off is what makes this safe without a mutex.
std::vector<float> g_last_recording;

PythonBridge g_bridge;

// --- WAV FILE UTILITY ---
// Writes mono 16-bit PCM. Header fields are written in host byte order, which is little-endian on x86 and ARM, as WAV requires.
bool SaveWavFile(const std::vector<float>& samples, const std::string& filename, int sampleRate) {
    if (samples.empty()) return false;
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    const uint16_t audioFormat = 1; // PCM
    const uint16_t channels = 1;
    const uint16_t bitsPerSample = 16;
    const uint16_t blockAlign = channels * bitsPerSample / 8;
    const uint32_t rate = static_cast<uint32_t>(sampleRate);
    const uint32_t byteRate = rate * blockAlign;
    const uint32_t dataSize = static_cast<uint32_t>(samples.size() * blockAlign);
    const uint32_t riffSize = 36 + dataSize;
    const uint32_t fmtSize = 16;

    auto put = [&file](const auto& value) { file.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
    file.write("RIFF", 4); put(riffSize); file.write("WAVE", 4);
    file.write("fmt ", 4); put(fmtSize); put(audioFormat); put(channels);
    put(rate); put(byteRate); put(blockAlign); put(bitsPerSample);
    file.write("data", 4); put(dataSize);

    std::vector<int16_t> pcm(samples.size());
    std::transform(samples.begin(), samples.end(), pcm.begin(), [](float s) {
        return static_cast<int16_t>(std::clamp(s, -1.0f, 1.0f) * 32767.0f);
    });
    file.write(reinterpret_cast<const char*>(pcm.data()), dataSize);
    return file.good();
}

// --- AUDIO CALLBACK ---
// Runs on miniaudio's real-time thread: no allocation, no locks, no I/O.
void data_callback(ma_device*, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    g_rb_input.Write(static_cast<const float*>(pInput), frameCount); // Drops samples if the worker falls behind.

    float* out = static_cast<float*>(pOutput);
    size_t played = g_rb_output.Read(out, frameCount);
    std::fill(out + played, out + frameCount, 0.0f); // Silence when nothing is queued.
}

// --- WORKER THREAD ---
// Peak-normalize to 0.9 so quiet mics still give the model a usable level.
// Gain is capped at 5x so near-silence is not boosted into loud noise.
void Normalize(std::vector<float>& clip) {
    float peak = 0.0f;
    for (float s : clip) peak = std::max(peak, std::abs(s));
    if (peak <= 0.001f) return;
    float gain = std::min(0.9f / peak, 5.0f);
    for (float& s : clip) s *= gain;
}

void processing_thread_func() {
    constexpr size_t kChunk = 1024;
    std::vector<float> chunk(kChunk);
    std::vector<float> clip;
    clip.reserve(static_cast<size_t>(kMaxRecordSeconds * kSampleRate) + kChunk);

    while (g_running) {
        if (g_rb_input.AvailableRead() < kChunk) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        g_rb_input.Read(chunk.data(), kChunk);

        // The mic runs all the time; drop its audio unless a recording was requested.
        if (!g_is_recording) {
            clip.clear();
            continue;
        }

        clip.insert(clip.end(), chunk.begin(), chunk.end());
        g_recorded_samples = clip.size();
        if (clip.size() < static_cast<size_t>(g_config.recording_duration * kSampleRate)) continue;

        g_is_processing = true;
        g_last_voice_activity = g_denoise ? Denoise(clip) : -1.0f;
        Normalize(clip);

        ConvertedAudio reply = g_bridge.ProcessAudio(Resample(clip, kSampleRate, kServerSampleRate),
                                                     static_cast<int>(std::lround(g_config.pitch_shift_semitones)));
        std::vector<float> converted = Resample(reply.samples, reply.sample_rate, kSampleRate);

        // Sample-count correction. The model emits audio in whole frames (10 ms each), so its output
        // can be a few ms shorter or longer than what was recorded. Stretch it by that tiny ratio so
        // the result lines up exactly with the recording. A large mismatch means something upstream
        // is broken (e.g. the model running at the wrong frame rate), so it is reported, not hidden.
        long drift = static_cast<long>(converted.size()) - static_cast<long>(clip.size());
        bool correctable = !converted.empty() && std::abs(drift) <= kMaxDriftFraction * clip.size();
        if (correctable) converted = ResampleToLength(converted, clip.size());
        g_last_drift = drift;
        g_last_drift_corrected = correctable;

        g_last_ok = !converted.empty();
        if (g_last_ok) {
            g_rb_output.Write(converted.data(), converted.size());
            g_last_recording = std::move(converted);
        }

        clip.clear();
        g_is_processing = false;
        g_is_recording = false; // Hands g_last_recording back to the main thread.
    }
}

// --- COMMANDS (main thread) ---
// Parses the whole string as a finite float. Rejects "abc", "5abc", "inf" and "nan".
bool ParseFloat(const std::string& text, float& out) {
    try {
        size_t used = 0;
        out = std::stof(text, &used);
        return used == text.size() && std::isfinite(out);
    } catch (const std::exception&) {
        return false;
    }
}

void WaitForPlayback() {
    while (g_rb_output.AvailableRead() > 0) std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

void RecordAndConvert(float seconds) {
    g_config.recording_duration = seconds;
    g_recorded_samples = 0;
    g_is_recording = true;

    const char spinner[] = {'|', '/', '-', '\\'};
    int spin = 0;
    while (g_is_recording) {
        if (g_is_processing) {
            std::cout << "\r[PROCESSING] Sending audio to AI... " << spinner[spin++ % 4] << "   " << std::flush;
        } else {
            std::cout << "\r[REC] " << std::fixed << std::setprecision(1)
                      << static_cast<float>(g_recorded_samples) / kSampleRate << "s / " << seconds << "s   " << std::flush;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!g_last_ok) {
        std::cout << "\r[PROCESSING] Sending audio to AI... [ERROR]" << std::endl;
        return;
    }
    std::cout << "\r[PROCESSING] Sending audio to AI... [DONE] " << std::endl;
    if (g_last_voice_activity >= 0.0f)
        std::cout << "[DENOISE] RNNoise speech probability: " << std::setprecision(2) << g_last_voice_activity << std::endl;
    double driftMs = g_last_drift * 1000.0 / kSampleRate;
    if (g_last_drift_corrected)
        std::cout << "[SYNC] Corrected " << g_last_drift << " samples (" << std::setprecision(1) << driftMs << " ms) of drift" << std::endl;
    else
        std::cout << "[WARNING] Output length is off by " << std::setprecision(1) << driftMs
                  << " ms, too much to be drift; playing it uncorrected" << std::endl;
    std::cout << "[PLAYING AI RESULT]" << std::endl;
    WaitForPlayback();
}

void Replay() {
    if (g_last_recording.empty()) {
        std::cout << "[ERROR] Nothing to replay yet. Record a clip first." << std::endl;
        return;
    }
    std::cout << "[REPLAYING]..." << std::endl;
    g_rb_output.Write(g_last_recording.data(), g_last_recording.size()); // The worker is idle, so main is the only writer.
    WaitForPlayback();
}

void SelectPitch() {
    const int maxPitch = static_cast<int>(kMaxPitchSemitones);
    std::cout << "Enter new pitch in semitones (" << -maxPitch << " to " << maxPitch << "): ";
    std::string line;
    float pitch = 0.0f;
    if (!std::getline(std::cin, line) || !ParseFloat(line, pitch) || std::abs(pitch) > kMaxPitchSemitones) {
        std::cout << "[ERROR] Invalid pitch, keeping " << g_config.pitch_shift_semitones << "." << std::endl;
        return;
    }
    g_config.pitch_shift_semitones = pitch;
    std::cout << "Pitch set to " << pitch << " semitones." << std::endl;
}

void SaveLastClip() {
    if (g_last_recording.empty()) {
        std::cout << "[ERROR] No audio clip to save." << std::endl;
        return;
    }

    std::string name;
    std::cout << "Enter filename (without extension): ";
    if (!std::getline(std::cin, name) || name.empty()) {
        std::time_t t = std::time(nullptr);
        std::tm tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << "output_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
        name = oss.str();
    }

    std::error_code ec;
    std::filesystem::create_directories(kSaveDir, ec);
    std::string path = std::string(kSaveDir) + "/" + name + ".wav";

    if (!ec && SaveWavFile(g_last_recording, path, kSampleRate))
        std::cout << "[SUCCESS] Saved to " << path << std::endl;
    else
        std::cout << "[ERROR] Failed to save " << path << std::endl;
}

// Blocks with a progress bar until the Python server accepts the connection.
void WaitForServer() {
    using namespace indicators;
    IndeterminateProgressBar server_bar{
        option::BarWidth{30},
        option::Start{"["},
        option::Fill{"·"},
        option::Lead{"<==>"},
        option::End{"]"},
        option::PrefixText{"Waiting for Server"},
        option::ForegroundColor{Color::yellow},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    };

    show_console_cursor(false);
    while (!g_bridge.Connect()) {
        server_bar.tick();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    server_bar.mark_as_completed();
    show_console_cursor(true);
    std::cout << termcolor::bold << termcolor::green << "Server connected!\n" << termcolor::reset;
}

// --- MAIN FUNCTION ---
int main() {
    std::cout << "========================================\n";
    std::cout << "                RVC CLIENT              \n";
    std::cout << "========================================\n";

    WaitForServer();

    g_rb_input.Init(kRingBufferSamples);
    g_rb_output.Init(kRingBufferSamples);

    ma_device_config config = ma_device_config_init(ma_device_type_duplex);
    config.sampleRate = kSampleRate;
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.playback.format = ma_format_f32;
    config.playback.channels = 1;
    config.dataCallback = data_callback;
    config.periodSizeInFrames = 4096; // A hint only; the callback handles any frameCount.
    config.periods = 3;

    ma_device device;
    if (ma_device_init(NULL, &config, &device) != MA_SUCCESS) {
        std::cerr << "[ERROR] Could not open the audio device." << std::endl;
        return 1;
    }
    if (ma_device_start(&device) != MA_SUCCESS) {
        std::cerr << "[ERROR] Could not start the audio device." << std::endl;
        ma_device_uninit(&device);
        return 1;
    }
    std::thread worker(processing_thread_func);

    while (true) {
        std::cout << "\n----------------------------------------\n";
        std::cout << "### Enter seconds to record (0.1-" << kMaxRecordSeconds << ") or a command:\n";
        std::cout << "### (-1: Replay, -2: Select Pitch, 's': Save Last Clip, 'd': Denoise "
                  << (g_denoise ? "ON" : "OFF") << ", 'q': Quit)\n> ";

        std::string line;
        if (!std::getline(std::cin, line) || line == "q") break; // EOF (Ctrl-D) quits too.
        if (line.empty()) continue;
        if (line == "s") { SaveLastClip(); continue; }
        if (line == "d") {
            g_denoise = !g_denoise;
            std::cout << "RNNoise denoising " << (g_denoise ? "enabled." : "disabled.") << std::endl;
            continue;
        }

        float value = 0.0f;
        if (!ParseFloat(line, value)) {
            std::cout << "[ERROR] Invalid input. Enter seconds, 's', -1, -2 or 'q'." << std::endl;
        } else if (value == -1.0f) {
            Replay();
        } else if (value == -2.0f) {
            SelectPitch();
        } else if (value >= 0.1f && value <= kMaxRecordSeconds) {
            RecordAndConvert(value);
        } else {
            std::cout << "[ERROR] Recording length must be between 0.1 and " << kMaxRecordSeconds << " seconds." << std::endl;
        }
    }

    // --- CLEANUP AND SHUTDOWN ---
    g_running = false;
    worker.join();
    ma_device_uninit(&device);
    g_bridge.Disconnect();

    std::cout << "[Client] Shutdown complete. Exiting." << std::endl;
    return 0;
}
