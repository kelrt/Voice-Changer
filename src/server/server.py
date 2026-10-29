# server.py
#
# Inference server for the C++ client (see ARCHITECTURE.md). Receives 16 kHz mono clips over TCP
# and returns them converted to the target voice, at the voice model's own sample rate. All
# resampling to and from 48 kHz happens in the client.
#
#   python src/server/server.py                                        serve the C++ client
#   python src/server/server.py --selftest in.wav out.wav [--pitch 12]  convert a file, no client
#
# Pipeline (RVC v2):
#   ContentVec (ONNX)  16 kHz audio -> 768-dim "what is being said" features, 50 frames/s
#   torchcrepe         16 kHz audio -> pitch track, 100 frames/s, shifted by N semitones
#   Synthesizer (PyTorch, Applio's RVC v2 code) features + pitch -> audio in the target voice
import argparse
import configparser
import socket
import struct
import sys
import time
import traceback
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchcrepe

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
from rvc.lib.algorithm.synthesizers import Synthesizer  # noqa: E402  (needs the vendor path above)

INPUT_SR = 16000          # The rate the client sends and both analysis models expect.
HOP = 160                 # 10 ms at 16 kHz: one pitch frame, and one synthesizer frame after upsampling features.
F0_MIN, F0_MAX = 50, 1100 # Range of the coarse pitch embedding the voice model was trained with.
VOICED_THRESHOLD = 0.21   # torchcrepe periodicity below this is treated as unvoiced (breath, consonants, silence).
# Measured on a 5 s clip (CPU): 'tiny' takes 1.4 s vs 15.9 s for 'full', agrees on 98% of
# voiced/unvoiced decisions, and differs by a median 12 cents (1/8 semitone). Not worth 11x the time.
CREPE_MODEL = 'tiny'

GREEN = "\033[32m"
RESET = "\033[0m"


def log(msg):
    print(f"{GREEN}{msg}{RESET}", flush=True)


# ---------------- MODELS ----------------
class VoiceConverter:
    def __init__(self, contentvec_path, model_path):
        for path in (contentvec_path, model_path):
            if not path.is_file():
                raise SystemExit(f"[Server] Missing model file {path}. Run scripts/fetch_models.sh "
                                 f"and put your RVC checkpoint at models/voice.pth.")

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.contentvec = ort.InferenceSession(str(contentvec_path), opts)
        log(f"[Server] ContentVec loaded from {contentvec_path}")

        ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
        if ckpt.get('version') != 'v2' or not ckpt.get('f0'):
            raise SystemExit(f"[Server] {model_path} is not an RVC v2 model with pitch guidance")
        self.sample_rate = int(ckpt['sr'])
        self.net = Synthesizer(*ckpt['config'], use_f0=True, text_enc_hidden_dim=768,
                               vocoder=ckpt.get('vocoder', 'HiFi-GAN'))
        # enc_q (the posterior encoder) is only used in training and is stripped from inference
        # checkpoints. Everything else must load exactly: a silent partial load produces noise.
        del self.net.enc_q
        self.net.load_state_dict(ckpt['weight'], strict=True)
        self.net.eval()
        log(f"[Server] Voice model loaded from {model_path}: "
            f"'{ckpt.get('model_name', '?')}', RVC v2, {self.sample_rate} Hz")

    def features(self, audio_16k):
        embed = self.contentvec.run(['embed'], {'source': audio_16k[np.newaxis, np.newaxis, :]})[0]
        feats = torch.from_numpy(embed)  # [1, T, 768] at 50 frames/s
        # The synthesizer works at 100 frames/s (one frame per 10 ms hop), so every feature frame
        # is repeated. Skipping this halves the output length, which is the bug that made earlier
        # versions of this project sound sped up.
        return F.interpolate(feats.transpose(1, 2), scale_factor=2).transpose(1, 2)

    @staticmethod
    def pitch(audio_16k, semitones):
        f0, periodicity = torchcrepe.predict(
            torch.from_numpy(audio_16k).unsqueeze(0), INPUT_SR, hop_length=HOP,
            fmin=F0_MIN, fmax=F0_MAX, model=CREPE_MODEL, batch_size=512, device='cpu',
            decoder=torchcrepe.decode.viterbi, return_periodicity=True)
        periodicity = torchcrepe.filter.median(periodicity, 3)
        f0 = torchcrepe.threshold.At(VOICED_THRESHOLD)(f0, periodicity)  # Unvoiced frames become NaN.
        f0 = torch.nan_to_num(f0, nan=0.0).squeeze(0).numpy()
        return f0 * 2 ** (semitones / 12.0)

    @staticmethod
    def coarse_pitch(f0):
        # Quantize Hz onto the 1..255 mel-spaced bins of the model's pitch embedding (RVC's formula).
        # Unvoiced frames (f0 = 0) land in bin 1.
        mel = 1127 * np.log(1 + f0 / 700)
        mel_min, mel_max = 1127 * np.log(1 + F0_MIN / 700), 1127 * np.log(1 + F0_MAX / 700)
        voiced = mel > 0
        mel[voiced] = (mel[voiced] - mel_min) * 254 / (mel_max - mel_min) + 1
        return np.rint(np.clip(mel, 1, 255)).astype(np.int64)

    def convert(self, audio_16k, semitones):
        timings = {}
        t = time.time()
        feats = self.features(audio_16k)
        timings['contentvec'] = time.time() - t

        t = time.time()
        f0 = self.pitch(audio_16k, semitones)
        timings['crepe'] = time.time() - t

        frames = min(len(audio_16k) // HOP, feats.shape[1], len(f0))
        feats, f0 = feats[:, :frames], f0[:frames]

        t = time.time()
        with torch.no_grad():
            audio = self.net.infer(
                feats, torch.tensor([frames]),
                torch.from_numpy(self.coarse_pitch(f0.copy())).unsqueeze(0),
                torch.from_numpy(f0.astype(np.float32)).unsqueeze(0),
                torch.tensor([0]))[0]
        timings['synth'] = time.time() - t

        voiced = np.count_nonzero(f0) / max(frames, 1)
        log("[Server] " + ", ".join(f"{k} {v:.2f}s" for k, v in timings.items()) + f", {voiced:.0%} voiced")
        return np.clip(audio.squeeze().numpy(), -1.0, 1.0).astype(np.float32)


# ---------------- NETWORK ----------------
def recvall(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def serve(converter, host, port):
    log(f"[Server] Server ready. Listening on {host}:{port}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        while True:
            conn, addr = server.accept()
            log(f"[Server] Client connected: {addr}")
            with conn:
                while (header := recvall(conn, 8)) is not None:
                    count, pitch = struct.unpack('ii', header)
                    payload = recvall(conn, count * 4)
                    if payload is None:
                        break
                    audio = np.frombuffer(payload, dtype=np.float32).copy()  # frombuffer is read-only; torch needs writable.
                    log(f"[Server] Received {count / INPUT_SR:.2f}s of audio, pitch {pitch:+d}")
                    try:
                        out = converter.convert(audio, pitch)
                        conn.sendall(struct.pack('ii', len(out), converter.sample_rate) + out.tobytes())
                    except Exception:
                        print("[Server] Conversion failed, replying with 0 samples:")
                        traceback.print_exc()
                        conn.sendall(struct.pack('ii', 0, 0))
            log("[Server] Client disconnected")


# ---------------- SELF-TEST ----------------
def read_wav_16k(path):
    import soxr
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise SystemExit("[Server] --selftest expects a 16-bit PCM WAV")
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
        return soxr.resample(audio, w.getframerate(), INPUT_SR).astype(np.float32)


def write_wav(path, audio, sample_rate):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--selftest', nargs=2, metavar=('IN_WAV', 'OUT_WAV'),
                        help='convert a WAV file and exit, without the C++ client')
    parser.add_argument('--pitch', type=int, default=12, help='semitones for --selftest (default +12)')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    if not config.read(REPO_ROOT / 'config.ini'):
        raise SystemExit(f"[Server] Config not found: {REPO_ROOT / 'config.ini'}")
    converter = VoiceConverter(REPO_ROOT / config['PATHS']['CONTENTVEC_PATH'],
                               REPO_ROOT / config['PATHS']['MODEL_PATH'])

    if args.selftest:
        audio = read_wav_16k(args.selftest[0])
        out = converter.convert(audio, args.pitch)
        write_wav(args.selftest[1], out, converter.sample_rate)
        log(f"[Server] {len(audio) / INPUT_SR:.2f}s in -> {len(out) / converter.sample_rate:.2f}s out, "
            f"written to {args.selftest[1]}")
        return

    try:
        serve(converter, config['NETWORK'].get('HOST', '127.0.0.1'), config['NETWORK'].getint('PORT', 51235))
    except KeyboardInterrupt:
        log("[Server] Server terminated")


if __name__ == '__main__':
    main()
