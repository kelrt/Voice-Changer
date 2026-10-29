#!/bin/bash
# Downloads the pretrained models server.py needs into models/ and verifies their SHA-256,
# so a truncated or corrupted download fails loudly instead of crashing the server later.
#
# The voice model itself (models/voice.pth) is not downloaded: it is your own trained RVC
# checkpoint. Put it there by hand.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

fetch() {
    local url="$1" dest="$2" sha256="$3"
    if [[ -f "$dest" ]] && echo "$sha256  $dest" | sha256sum --check --status; then
        echo "[ok]       $dest"
        return
    fi
    echo "[download] $dest"
    curl -fL --retry 3 -o "$dest.part" "$url"
    if ! echo "$sha256  $dest.part" | sha256sum --check --status; then
        echo "[error]    checksum mismatch for $dest, keeping $dest.part for inspection" >&2
        exit 1
    fi
    mv "$dest.part" "$dest"
    echo "[ok]       $dest"
}

# ContentVec (HuBERT architecture), layer-12 768-dim features, as used by RVC v2 models.
fetch "https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/resolve/main/vec-768-layer-12.onnx" \
      models/contentvec.onnx \
      b3886e7dff1495cda514f94f4680a7b1261e05d6929f5c764cdb17934b413c2a

[[ -f models/voice.pth ]] || echo "[missing]  models/voice.pth (your RVC v2 checkpoint)" >&2
