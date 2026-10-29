// /src/PythonBridge.h

#ifndef VOICE_CHANGER_PYTHONBRIDGE_H
#define VOICE_CHANGER_PYTHONBRIDGE_H

#include <cstddef>
#include <string>
#include <vector>

// Audio returned by the server, at whatever rate the voice model produces.
struct ConvertedAudio {
    std::vector<float> samples;
    int sample_rate = 0;
};

// TCP client for the Python inference server (src/server/server.py).
//
// Wire format (native byte order, both ends run on the same machine):
//   request:  int32 sample_count, int32 pitch_semitones, float32[sample_count]                mono, 16 kHz
//   response: int32 sample_count, int32 sample_rate, float32[sample_count]                    mono, model rate
// The client does all resampling: it sends 16 kHz (the rate the models analyze at) and converts
// the reply from the model's rate back to 48 kHz. A response count of 0 means the server failed.
class PythonBridge {
public:
    explicit PythonBridge(std::string host = "127.0.0.1", int port = 51235);
    ~PythonBridge();

    // Owns a socket, so copying would double-close it.
    PythonBridge(const PythonBridge&) = delete;
    PythonBridge& operator=(const PythonBridge&) = delete;

    // Returns false if the server is not reachable yet. Safe to call repeatedly.
    bool Connect();
    void Disconnect();

    // Sends one clip and blocks until the converted clip comes back.
    // Reconnects once if the connection was lost earlier. Returns empty samples on any failure.
    ConvertedAudio ProcessAudio(const std::vector<float>& input16k, int pitch_semitones);

private:
    bool SendAll(const void* data, size_t len);
    bool RecvAll(void* data, size_t len);

    std::string host;
    int port;
    int sock = -1; // -1 means disconnected.
};

#endif
