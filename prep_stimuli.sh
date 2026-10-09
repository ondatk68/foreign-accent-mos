#!/usr/bin/env bash
# Run from any directory; SCP paths are relative to the repository root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python_bin="${PYTHON:-python3}"
bigvgan_dir="${BIGVGAN_DIR:-external/BigVGAN}"
sidon_dir="${SIDON_DIR:-external/Sidon}"
device="${DEVICE:-cuda}"
if [[ ! -f "$sidon_dir/src/sidon/cleansing/__init__.py" ]]; then
    echo "Set SIDON_DIR to a sarulab-speech/Sidon source checkout containing src/sidon/cleansing/__init__.py." >&2
    exit 1
fi
if [[ ! -f "$bigvgan_dir/bigvgan.py" ]]; then
    echo "Set BIGVGAN_DIR to an NVIDIA BigVGAN source checkout." >&2
    exit 1
fi
"$python_bin" scripts/prep_sidon.py --device "$device" --sidon-dir "$sidon_dir"
"$python_bin" scripts/prep_bigvgan.py --device "$device" --bigvgan-dir "$bigvgan_dir"
"$python_bin" scripts/prep_encodec.py --device "$device"
"$python_bin" scripts/prep_world.py
cat data/NoResyn.scp data/BigVGAN.scp data/EnCodec.scp data/WORLD.scp > stimuli.scp
