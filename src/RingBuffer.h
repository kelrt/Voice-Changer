// /src/RingBuffer.h

#ifndef VOICE_CHANGER_RINGBUFFER_H
#define VOICE_CHANGER_RINGBUFFER_H

#include <algorithm>
#include <atomic>
#include <vector>

// --- RING BUFFER ---
// Lock-free single-producer / single-consumer queue of audio samples.
// Exactly one thread may call Write and exactly one thread may call Read at any time.
// The indices only ever grow; (write - read) is the fill level, and index % capacity is the slot.
class AudioRingBuffer {
private:
    std::vector<float> m_buffer;
    std::atomic<size_t> m_writeIndex{0}; // Total samples ever written.
    std::atomic<size_t> m_readIndex{0};  // Total samples ever read.

public:
    // Allocates the storage. Call once, before any thread uses the buffer.
    void Init(size_t capacity) { m_buffer.assign(capacity, 0.0f); }

    size_t AvailableRead() const { return m_writeIndex.load() - m_readIndex.load(); }
    size_t AvailableWrite() const { return m_buffer.size() - AvailableRead(); }

    // Writes up to `count` samples and returns how many fit. Never overwrites unread data.
    size_t Write(const float* data, size_t count) {
        count = std::min(count, AvailableWrite());
        size_t w = m_writeIndex.load();
        for (size_t i = 0; i < count; ++i) m_buffer[(w + i) % m_buffer.size()] = data[i];
        m_writeIndex.store(w + count); // Publish only after the samples are in place.
        return count;
    }

    // Reads up to `count` samples and returns how many were available.
    size_t Read(float* out, size_t count) {
        count = std::min(count, AvailableRead());
        size_t r = m_readIndex.load();
        for (size_t i = 0; i < count; ++i) out[i] = m_buffer[(r + i) % m_buffer.size()];
        m_readIndex.store(r + count);
        return count;
    }
};

#endif
