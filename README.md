# Voice-Changer

A voice converter that records a clip from your microphone and re-speaks it in another voice using an RVC (Retrieval-based Voice Conversion) model. It's built for quality over latency.

- The **C++ client** handles audio I/O, RNNoise denoising, band-limited resampling and sample-count correction.
- A **Python server** runs the neural networks: ContentVec, torchcrepe and an RVC v2 synthesizer.

## Demo
**Input**
asdfasdfasdf

**Output**
asdfasdfadsf


## Setup

**Server** (Python 3.10):
```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
scripts/fetch_models.sh          # downloads ContentVec and verifies its checksum
cp /path/to/your_model.pth models/voice.pth   # your RVC v2 checkpoint (pitch-guided)
src/server/start_server.sh
```

**Client** (Linux; needs CMake 3.20+, a C++17 compiler and RNNoise, e.g. `dnf install rnnoise-devel`):
```sh
cmake -S . -B build && cmake --build build
./build/voice_changer_app
```

The client waits until the server is up before prompting for input. Once the server is connected, the following options are available:

| Input | Action |
|---|---|
| `-1` | Replay |
| `-2` | Change pitch (+12 for male → female) |
| `d` | Toggle denoising |
| `s` | Save the result as a WAV |
| `q` | Quit |

**Without models:**
- `python src/server/mock_server.py` stands in for the server, so you can try the client.
- `python src/server/server.py --selftest in.wav out.wav --pitch 12` converts a file without the client.

## Third-party code

- [miniaudio](https://github.com/mackron/miniaudio): audio I/O 
- [indicators](https://github.com/p-ranav/indicators): terminal progress bar(i spent wayyy too much time on this...)
- [RNNoise](https://github.com/xiph/rnnoise): noise suppression
- [Applio](https://github.com/IAHispano/Applio): RVC v2 model definition
- ContentVec ONNX export from [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel)
