// /src/Denoiser.h

#ifndef VOICE_CHANGER_DENOISER_H
#define VOICE_CHANGER_DENOISER_H

#include <vector>

// Removes steady background noise (fans, hum, keyboard) from a 48 kHz mono clip, in place,
// using RNNoise: a small recurrent neural network from Xiph that estimates per-band gains
// for each 10 ms frame. Returns the average voice-activity probability RNNoise reported (0-1).
float Denoise(std::vector<float>& clip);

#endif
