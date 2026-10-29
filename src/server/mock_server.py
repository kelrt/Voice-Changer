# mock_server.py
#
# Stand-in for server.py that speaks the same wire protocol but runs no models. It echoes the
# client's audio back the way an RVC model would shape it: at the model's sample rate and in
# whole 10 ms frames, so the reply is slightly shorter than the request. This exercises the
# client's resampling and sample-count correction without needing any model files.
#
#   python src/server/mock_server.py [--rate 32000] [--port 51235]
import argparse
import socket
import struct

import numpy as np

INPUT_SR = 16000
HOP = 160  # 10 ms at 16 kHz: RVC produces one frame of audio per hop.


def recvall(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def fake_convert(audio_16k, model_sr):
    frames = len(audio_16k) // HOP                     # Whole frames only, like the real model.
    out_len = frames * model_sr // 100
    t_out = np.arange(out_len) / model_sr
    t_in = np.arange(len(audio_16k)) / INPUT_SR
    return np.interp(t_out, t_in, audio_16k).astype(np.float32) * 0.8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate', type=int, default=32000, help='sample rate to reply at (the "model" rate)')
    parser.add_argument('--port', type=int, default=51235)
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', args.port))
        server.listen()
        print(f"[Mock] Listening on 127.0.0.1:{args.port}, replying at {args.rate} Hz")
        while True:
            conn, _ = server.accept()
            print("[Mock] Client connected")
            with conn:
                while (header := recvall(conn, 8)) is not None:
                    count, pitch = struct.unpack('ii', header)
                    payload = recvall(conn, count * 4)
                    if payload is None:
                        break
                    out = fake_convert(np.frombuffer(payload, dtype=np.float32), args.rate)
                    print(f"[Mock] {count} samples in (pitch {pitch:+d}) -> {len(out)} samples at {args.rate} Hz")
                    conn.sendall(struct.pack('ii', len(out), args.rate) + out.tobytes())
            print("[Mock] Client disconnected")


if __name__ == '__main__':
    main()
