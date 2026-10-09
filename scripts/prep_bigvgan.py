"""Resynthesize denoised speech with BigVGAN v2 at 44.1 kHz."""
import sys
from stimuli_utils import arguments, output_path, save_stimulus


def main():
    args = arguments("BigVGAN")
    sys.path.insert(0, str(args.bigvgan_dir.resolve()))
    import bigvgan
    import librosa
    import torch
    from meldataset import get_mel_spectrogram
    from tqdm import tqdm

    model = bigvgan.BigVGAN.from_pretrained(
        "nvidia/bigvgan_v2_44khz_128band_512x", use_cuda_kernel=False
    )
    model.remove_weight_norm()
    model = model.eval().to(args.device)
    with args.output_scp.open("w") as scp, torch.inference_mode():
        for key, path in tqdm(args.entries, desc="BigVGAN"):
            wav, _ = librosa.load(path, sr=model.h.sampling_rate, mono=True)
            mel = get_mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), model.h)
            audio = model(mel.to(args.device)).reshape(-1).cpu().numpy()
            key, target = output_path(args, key, "BigVGAN")
            save_stimulus(target, audio, model.h.sampling_rate)
            scp.write(f"{key} {target.as_posix()}\n")


if __name__ == "__main__":
    main()
