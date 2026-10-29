# Architecture

The voice changer works on recorded clips, not a live stream: quality matters more than latency here. You record a clip of a set length, a Python server runs RVC (Retrieval-based Voice Conversion) inference on it, and the converted voice plays back.

The split between the two processes:
- The **C++ client** owns the audio hardware, the threading, every bit of signal processing (denoising and all sample-rate conversion), and the command prompt.
- The **Python server** only runs the neural networks.

They talk over a local TCP socket.

```
            C++ client                                                Python server (src/server/server.py)
 mic ─▶ audio callback ─▶ g_rb_input ─▶ worker thread
                                         RNNoise denoise (48 kHz)
                                         peak normalize
                                         resample 48 → 16 kHz ─── TCP ───▶ ContentVec (ONNX): content features
                                                                           torchcrepe: pitch track, shifted +N semitones
                                                                           RVC v2 synthesizer (PyTorch): target voice
                                         resample model rate → 48 kHz ◀─── TCP ─── audio at the model's rate (32 kHz)
                                         sample-count correction
 speakers ◀── audio callback ◀── g_rb_output ◀┘
```

## C++ client

| File | Responsibility |
|---|---|
| `src/main.cpp` | Audio device, threads, clip pipeline, command prompt, WAV saving |
| `src/RingBuffer.h` | `AudioRingBuffer`: lock-free single-producer/single-consumer queue |
| `src/Denoiser.{h,cpp}` | RNNoise wrapper |
| `src/Resampler.{h,cpp}` | Windowed-sinc sample-rate converter |
| `src/PythonBridge.{h,cpp}` | TCP client and wire protocol |

**Threads.** All client audio is mono, float, 48 kHz. miniaudio converts to and from the hardware's own format and channel count.

| Thread | Does | Must not |
|---|---|---|
| Audio callback (miniaudio) | Writes mic samples into `g_rb_input`. Reads `g_rb_output` into the speakers and fills any shortfall with silence. | Block, allocate, or do I/O |
| Worker (`processing_thread_func`) | Drains `g_rb_input` in 1024-sample chunks and discards them unless a recording is in progress. Once the clip is long enough, runs the clip pipeline below. | Write to the console |
| Main | Command prompt, progress and spinner output, replay, WAV save | Touch audio while the worker is busy |

**Clip pipeline** (worker thread):
1. **Denoise.** RNNoise is Xiph's small recurrent network that estimates per-band gains for each 10 ms frame (480 samples). It removes steady noise like fans and hum before the models hear it. Each clip gets a fresh state, so a new room's noise isn't judged by the old one's profile. The `d` command toggles it for A/B comparison.
2. **Normalize.** Scale the peak to 0.9, with gain capped at 5×, so quiet mics still give the models a usable level.
3. **Resample 48 → 16 kHz.** Both analysis models work at 16 kHz, so this is all the server needs, and it cuts the bytes sent by 3×.
4. **Server round trip.** Audio comes back at the voice model's own rate.
5. **Resample the model's rate → 48 kHz.**
6. **Sample-count correction.** The model emits audio in whole 10 ms frames, so the result can be a few milliseconds off the recording's length. If it's within 2%, the clip is stretched by that tiny ratio so it lines up exactly. A larger mismatch means a bug upstream: an earlier version of this project produced half-length output and silently stretched it back. That gets reported as a warning, never hidden.

**Resampler.** Each output sample is a weighted sum of nearby input samples. The weights come from a sinc low-pass filter shaped by a Kaiser window (32 zero crossings each side, β = 8.6, tabulated at 512 points per zero crossing). The cutoff sits at 95% of the lower rate's Nyquist frequency, so downsampling removes what the new rate can't represent instead of letting it alias.

Measured against libsoxr's very-high-quality mode on sine sweeps:

| Test | Result |
|---|---|
| 48 → 16 kHz, 32 → 48 kHz, 40 → 48 kHz | 90–101 dB SNR against soxr |
| Aliasing: 12 kHz tone downsampled to 16 kHz | −107 dB residual |
| Speed | About 55 ms per 5 s clip |

**Ring buffers** (`AudioRingBuffer`). Read and write positions only ever grow, so the fill level is `write - read`. `Write` and `Read` move only as many samples as there is room or data for, so unread audio is never overwritten.

**Thread handoff.** Main sets `g_is_recording`, and the worker clears it after storing the result in `g_last_recording`. Only the side that currently owns that flag touches `g_last_recording` and `g_rb_output`'s write side, so the atomic flag orders every access without a mutex.

**Commands:**

| Input | Action |
|---|---|
| `0.1`–`25` | Record that many seconds |
| `-1` | Replay the last result |
| `-2` | Set the pitch shift (±24 semitones; +12 is one octave, typical for male → female) |
| `d` | Toggle RNNoise |
| `s` | Save the last result to `saved_outputs/` |
| `q` or Ctrl-D | Quit |

Clips are capped at 25 s so they fit in the 30 s ring buffers.

## Wire protocol (`PythonBridge`)

Both processes run on the same machine and use native byte order.

```
request:  int32 sample_count | int32 pitch_semitones | float32[sample_count]                   mono, 16 kHz
response: int32 sample_count | int32 sample_rate     | float32[sample_count]                   mono, model rate
```

- A response count of `0` means the server failed, and the server logs why.
- Sends and receives loop until every byte is through.
- `send` uses `MSG_NOSIGNAL`, so a dead server returns an error instead of killing the client with SIGPIPE.
- The client rejects implausible counts and rates.
- After any connection error, the next request reconnects.

`src/server/mock_server.py` speaks the same protocol with no models. It echoes audio back at a model-like rate in whole frames, which exercises the client's resampling and drift correction.

## Python server

**Startup.** Load `config.ini`, ContentVec and the voice checkpoint.
- The checkpoint's own metadata supplies the sample rate (32 kHz for the current model) and version, so there's no hand-maintained setting to get wrong.
- The network is built with Applio's RVC v2 code (`src/server/vendor/`), the same architecture the model was trained with.
- Every weight must load (`strict=True`). An earlier version of this project used a placeholder architecture with a non-strict load, which loaded nothing and produced noise.

**Per request:**
1. **Content.** ContentVec (HuBERT architecture, layer-12 768-dim features, via ONNX Runtime) turns 16 kHz audio into features at 50 frames/s. The synthesizer works at 100 frames/s, so each feature frame is repeated. Skipping that halves the output length.
2. **Pitch.** torchcrepe (`tiny` model, Viterbi decoding) produces a pitch track at 100 frames/s. Frames with low periodicity (breath, consonants, silence) are marked unvoiced. The pitch is multiplied by 2^(semitones/12) and quantized onto the model's 255 mel-spaced pitch bins.
3. **Synthesis.** The RVC v2 synthesizer (text encoder → normalizing flow → HiFi-GAN NSF vocoder) generates the target voice.

For a 4 s clip on CPU this takes about 3.6 s: ContentVec 0.2 s, CREPE 1.4 s, synthesis 2 s.

`server.py --selftest in.wav out.wav` runs the same pipeline on a file, without the client.

## Design notes

- **Why a separate Python process?** The models come from the PyTorch ecosystem. Keeping them out of the C++ process means the audio callback never waits on them, and they can change without rebuilding the client. The C++ side owns everything that's signal processing rather than neural networks.
- **Why CREPE `tiny`?** On a 5 s clip it's 11× faster than `full` (1.4 s vs 15.9 s). It agrees on 98% of voiced/unvoiced decisions and differs by a median of 12 cents.
- **Known limitations:**
  - Clips are converted whole; there's no streaming.
  - The voice model runs in PyTorch, not ONNX.
  - The protocol has no version number or error codes.
  - The client is POSIX-only (Linux/macOS).
- **Next steps:**
  - Export the synthesizer to ONNX, verified numerically against PyTorch, so the server needs no PyTorch.
  - RMVPE pitch detection.
  - Chunked streaming with crossfades for live use.
