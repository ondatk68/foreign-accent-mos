"""Shared SCP naming and command-line options."""
import argparse
from pathlib import Path

SYSTEMS = ("NoResyn", "BigVGAN", "EnCodec", "WORLD")


def read_scp(path):
    entries = []
    seen = set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        key, audio = line.split(maxsplit=1)
        if key in seen or Path(key).name != key or key in (".", ".."):
            raise ValueError(f"Invalid or duplicate ID at {path}:{number}: {key}")
        seen.add(key)
        entries.append((key, Path(audio)))
    return entries


def condition_id(key, system):
    parts = key.split("_")
    if len(parts) < 3 or parts[-1] not in ("orig", *SYSTEMS):
        raise ValueError(f"Expected <speaker>_<sentence>_<condition>, got {key}")
    parts[-1] = system
    return "_".join(parts)


def arguments(system):
    parser = argparse.ArgumentParser(description=f"Prepare {system} stimuli.")
    parser.add_argument("--input-scp", default="original.scp" if system == "NoResyn" else "data/intermediate/NoResyn.scp")
    parser.add_argument("--output-dir", type=Path, default=Path("data") / system)
    parser.add_argument("--output-scp", type=Path, default=Path("data") / f"{system}.scp")
    if system != "WORLD":
        parser.add_argument("--device", default="cuda")
    if system == "NoResyn":
        parser.add_argument("--native-dir", type=Path, default=Path("data/intermediate/NoResyn"))
        parser.add_argument("--native-scp", type=Path, default=Path("data/intermediate/NoResyn.scp"))
        parser.add_argument("--sidon-dir", type=Path, default=Path("external/Sidon"), help="Path to a sarulab-speech/Sidon source checkout")
    if system == "BigVGAN":
        parser.add_argument("--bigvgan-dir", type=Path, required=True, help="Path to an NVIDIA BigVGAN source checkout")
    args = parser.parse_args()
    if system == "NoResyn" and not (args.sidon_dir / "src/sidon/cleansing/__init__.py").is_file():
        parser.error("Sidon checkout must contain src/sidon/cleansing/__init__.py; set --sidon-dir")
    args.entries = read_scp(args.input_scp)
    missing = [str(path) for _, path in args.entries if not path.is_file()]
    if missing:
        parser.error(f"Missing input audio: {missing[0]} ({len(missing)} missing files)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_scp.parent.mkdir(parents=True, exist_ok=True)
    return args


def output_path(args, key, system):
    key = condition_id(key, system)
    return key, args.output_dir / f"{key}.wav"


def save_stimulus(path, audio, sample_rate):
    """Resample a mono waveform to 16 kHz immediately before saving."""
    from math import gcd
    from scipy.signal import resample_poly
    import soundfile as sf

    if sample_rate != 16000:
        divisor = gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor)
    sf.write(path, audio, 16000, subtype="PCM_16")
