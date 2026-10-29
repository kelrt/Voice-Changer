// /src/Denoiser.cpp

#include "Denoiser.h"

#include <rnnoise.h>

#include <algorithm>
#include <memory>

float Denoise(std::vector<float>& clip) {
    // A fresh state per clip: RNNoise adapts its noise estimate within the first few frames,
    // and a clip recorded minutes later should not inherit the old room's noise profile.
    std::unique_ptr<DenoiseState, decltype(&rnnoise_destroy)> state(rnnoise_create(nullptr), &rnnoise_destroy);
    if (!state || clip.empty()) return 0.0f;

    const size_t frameSize = static_cast<size_t>(rnnoise_get_frame_size()); // 480 samples = 10 ms at 48 kHz.
    std::vector<float> frame(frameSize);
    double vadSum = 0.0;
    size_t frames = 0;

    for (size_t start = 0; start < clip.size(); start += frameSize) {
        size_t count = std::min(frameSize, clip.size() - start);
        // RNNoise expects 16-bit sample scale (+-32768), not +-1.0. The last frame is zero-padded.
        for (size_t i = 0; i < frameSize; ++i) frame[i] = i < count ? clip[start + i] * 32768.0f : 0.0f;
        vadSum += rnnoise_process_frame(state.get(), frame.data(), frame.data());
        for (size_t i = 0; i < count; ++i) clip[start + i] = frame[i] / 32768.0f;
        ++frames;
    }
    return static_cast<float>(vadSum / frames);
}
