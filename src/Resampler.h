// /src/Resampler.h

#ifndef VOICE_CHANGER_RESAMPLER_H
#define VOICE_CHANGER_RESAMPLER_H

#include <cstddef>
#include <vector>

// Band-limited (windowed-sinc) sample-rate conversion for mono float audio.
//
// Every output sample is a weighted sum of the input samples around its position, using a sinc
// low-pass filter shaped by a Kaiser window. The filter cutoff sits just below the Nyquist
// frequency of the lower of the two rates, so downsampling (e.g. 48 kHz -> 16 kHz) removes
// content the new rate cannot represent instead of letting it alias into audible noise.

// Converts between two sample rates, e.g. Resample(clip, 48000, 16000).
std::vector<float> Resample(const std::vector<float>& input, int srcRate, int dstRate);

// Resamples to exactly `outputLength` samples. Used for sample-count correction: stretching
// or squeezing a clip by a tiny ratio so its length matches the original recording exactly.
std::vector<float> ResampleToLength(const std::vector<float>& input, size_t outputLength);

#endif
