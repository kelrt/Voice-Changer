// /src/Resampler.cpp

#include "Resampler.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr int kZeroCrossings = 32;        // Filter half-width, in zero crossings of the sinc. More = sharper cutoff, slower.
constexpr int kTableResolution = 512;     // Table entries per zero crossing; values in between are linearly interpolated.
constexpr double kKaiserBeta = 8.6;       // Window shape: about 90 dB stopband attenuation.
constexpr double kRolloff = 0.95;         // Cutoff as a fraction of the lower Nyquist frequency, leaving room for the transition band.

// Zeroth-order modified Bessel function of the first kind, used by the Kaiser window.
double BesselI0(double x) {
    double sum = 1.0, term = 1.0;
    for (int k = 1; k < 50; ++k) {
        term *= (x / (2.0 * k)) * (x / (2.0 * k));
        sum += term;
        if (term < sum * 1e-12) break;
    }
    return sum;
}

// Windowed sinc evaluated at u = 0 .. kZeroCrossings (u in units of zero crossings).
// Computing it once avoids a sin() and a Bessel series for every filter tap.
const std::vector<float>& KernelTable() {
    static const std::vector<float> table = [] {
        std::vector<float> t(kZeroCrossings * kTableResolution + 2, 0.0f);
        const double i0Beta = BesselI0(kKaiserBeta);
        for (size_t i = 0; i < t.size(); ++i) {
            double u = static_cast<double>(i) / kTableResolution;
            if (u >= kZeroCrossings) break; // Window is zero at and beyond its edge.
            double sinc = (u == 0.0) ? 1.0 : std::sin(M_PI * u) / (M_PI * u);
            double r = u / kZeroCrossings;
            double window = BesselI0(kKaiserBeta * std::sqrt(1.0 - r * r)) / i0Beta;
            t[i] = static_cast<float>(sinc * window);
        }
        return t;
    }();
    return table;
}

float Kernel(double u) {
    const std::vector<float>& table = KernelTable();
    double pos = std::abs(u) * kTableResolution;
    size_t i = static_cast<size_t>(pos);
    if (i + 1 >= table.size()) return 0.0f;
    float frac = static_cast<float>(pos - i);
    return table[i] + (table[i + 1] - table[i]) * frac;
}

// ratio = output rate / input rate. Output sample n sits at input position n / ratio.
std::vector<float> ResampleByRatio(const std::vector<float>& input, double ratio, size_t outputLength) {
    std::vector<float> output(outputLength, 0.0f);
    if (input.empty() || outputLength == 0) return output;

    // When downsampling, the cutoff must drop to the output's Nyquist; scaling the kernel by
    // `cutoff` both lowers its frequency and keeps its gain at 1.
    const double cutoff = std::min(1.0, ratio) * kRolloff;
    const int halfWidth = static_cast<int>(std::ceil(kZeroCrossings / cutoff)); // In input samples.
    const long last = static_cast<long>(input.size()) - 1;

    for (size_t n = 0; n < outputLength; ++n) {
        double center = n / ratio;
        long first = std::max(0L, static_cast<long>(std::floor(center)) - halfWidth + 1);
        long end = std::min(last, static_cast<long>(std::floor(center)) + halfWidth);
        double acc = 0.0;
        for (long k = first; k <= end; ++k) acc += input[k] * Kernel((k - center) * cutoff);
        output[n] = static_cast<float>(acc * cutoff);
    }
    return output;
}

} // namespace

std::vector<float> Resample(const std::vector<float>& input, int srcRate, int dstRate) {
    if (srcRate == dstRate || input.empty()) return input;
    double ratio = static_cast<double>(dstRate) / srcRate;
    return ResampleByRatio(input, ratio, static_cast<size_t>(std::llround(input.size() * ratio)));
}

std::vector<float> ResampleToLength(const std::vector<float>& input, size_t outputLength) {
    if (input.size() == outputLength || input.empty()) return input;
    return ResampleByRatio(input, static_cast<double>(outputLength) / input.size(), outputLength);
}
