"""Denoise original speech with Sidon v0.1 (48 kHz intermediate, 16 kHz stimuli)."""
import sys
from stimuli_utils import arguments, output_path, save_stimulus


def main():
    args = arguments("NoResyn")
    sys.path.insert(0, str((args.sidon_dir / "src").resolve()))
    import soundfile as sf
    import torch
    import torchaudio
    from huggingface_hub import hf_hub_download
    from sidon.cleansing import extract_seamless_m4t_features
    from tqdm import tqdm

    torch.backends.cudnn.enabled = False
    models = []
    for filename in ("feature_extractor_cuda.pt", "decoder_cuda.pt"):
        path = hf_hub_download("sarulab-speech/sidon-v0.1", filename=filename)
        models.append(torch.jit.load(path, map_location=args.device).eval())
    extractor, decoder = models
    args.native_dir.mkdir(parents=True, exist_ok=True)
    args.native_scp.parent.mkdir(parents=True, exist_ok=True)
    with args.output_scp.open("w") as scp, args.native_scp.open("w") as native_scp, torch.inference_mode():
        for key, path in tqdm(args.entries, desc="Sidon"):
            wav, sr = torchaudio.load(path)
            wav = torchaudio.functional.resample(wav.mean(0), sr, 16000)
            if wav.numel() > 25 * 16000:
                raise ValueError(f"Sidon expects at most 25 seconds: {path}")
            padded = torch.nn.functional.pad(wav, (0, 25 * 16000 - wav.numel()))
            features = extract_seamless_m4t_features(
                [padded], return_tensors="pt", padding_value=1.0, device=args.device
            )
            hidden = extractor(features["input_features"].to(args.device))["last_hidden_state"]
            cleaned = decoder(hidden.transpose(1, 2)).reshape(-1).cpu()
            length = round(wav.numel() * 48000 / 16000)
            cleaned = torch.nn.functional.pad(cleaned, (0, max(0, length - cleaned.numel())))[:length]
            key, target = output_path(args, key, "NoResyn")
            native_target = args.native_dir / f"{key}.wav"
            sf.write(native_target, cleaned.numpy(), 48000, subtype="FLOAT")
            native_scp.write(f"{key} {native_target.as_posix()}\n")
            save_stimulus(target, cleaned.numpy(), 48000)
            scp.write(f"{key} {target.as_posix()}\n")


if __name__ == "__main__":
    main()
