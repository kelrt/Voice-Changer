// /src/PythonBridge.cpp

#include "PythonBridge.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <iostream>
#include <utility>

namespace {
// Two minutes of 48 kHz audio. A larger count means the stream is out of sync, not a real clip.
constexpr int32_t kMaxResponseSamples = 48000 * 120;
}

PythonBridge::PythonBridge(std::string host, int port) : host(std::move(host)), port(port) {}

PythonBridge::~PythonBridge() { Disconnect(); }

void PythonBridge::Disconnect() {
    if (sock >= 0) {
        close(sock);
        sock = -1;
    }
}

bool PythonBridge::Connect() {
    Disconnect();

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) return false;

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;

    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        Disconnect(); // Close the failed socket, or every retry leaks a file descriptor.
        return false;
    }
    return true;
}

// send() and recv() may move fewer bytes than asked, so both loop until done.
bool PythonBridge::SendAll(const void* data, size_t len) {
    const char* p = static_cast<const char*>(data);
    while (len > 0) {
        // MSG_NOSIGNAL: if the server died we want an error, not a SIGPIPE that kills the client.
        ssize_t n = send(sock, p, len, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

bool PythonBridge::RecvAll(void* data, size_t len) {
    char* p = static_cast<char*>(data);
    while (len > 0) {
        ssize_t n = recv(sock, p, len, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false; // 0 means the server closed the connection.
        p += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

ConvertedAudio PythonBridge::ProcessAudio(const std::vector<float>& input16k, int pitch_semitones) {
    if (sock < 0 && !Connect()) {
        std::cerr << "\n[Bridge] Server is not reachable." << std::endl;
        return {};
    }

    // Arrays of int32 rather than structs, so there is no padding to disagree with Python's 'ii'.
    int32_t request[2] = {static_cast<int32_t>(input16k.size()), static_cast<int32_t>(pitch_semitones)};
    int32_t response[2] = {0, 0}; // sample_count, sample_rate

    if (!SendAll(request, sizeof(request)) ||
        !SendAll(input16k.data(), input16k.size() * sizeof(float)) ||
        !RecvAll(response, sizeof(response))) {
        std::cerr << "\n[Bridge] Lost connection to the server." << std::endl;
        Disconnect();
        return {};
    }

    const int32_t count = response[0], rate = response[1];
    if (count == 0) return {};
    if (count < 0 || count > kMaxResponseSamples || rate < 8000 || rate > 192000) {
        std::cerr << "\n[Bridge] Invalid response header (" << count << " samples at " << rate
                  << " Hz), dropping connection." << std::endl;
        Disconnect();
        return {};
    }

    ConvertedAudio result;
    result.sample_rate = rate;
    result.samples.resize(static_cast<size_t>(count));
    if (!RecvAll(result.samples.data(), result.samples.size() * sizeof(float))) {
        std::cerr << "\n[Bridge] Connection dropped mid-response." << std::endl;
        Disconnect();
        return {};
    }
    return result;
}
