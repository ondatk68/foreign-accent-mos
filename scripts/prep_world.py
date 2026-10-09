"""Analyze and resynthesize denoised speech using WORLD."""
from stimuli_utils import arguments, output_path, save_stimulus


def main():
    args = arguments("WORLD")
    import numpy as np
    import pyworld as pw
    import soundfile as sf
    from tqdm import tqdm

    with args.output_scp.open("w") as scp:
        for key, path in tqdm(args.entries, desc="WORLD"):
            wav, sr = sf.read(path, always_2d=True)
            wav = np.ascontiguousarray(wav.mean(axis=1), dtype=np.float64)
            f0, sp, ap = pw.wav2world(wav, sr)
            audio = pw.synthesize(f0, sp, ap, sr)
            key, target = output_path(args, key, "WORLD")
            save_stimulus(target, audio, sr)
            scp.write(f"{key} {target.as_posix()}\n")


if __name__ == "__main__":
    main()
