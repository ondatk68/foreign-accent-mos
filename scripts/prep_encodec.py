"""Resynthesize denoised speech with EnCodec at 24 kHz, 12 kbps."""
from stimuli_utils import arguments, output_path, save_stimulus


def main():
    args = arguments("EnCodec")
    import torch
    import torchaudio
    from transformers import AutoProcessor, EncodecModel
    from tqdm import tqdm

    model = EncodecModel.from_pretrained("facebook/encodec_24khz").eval().to(args.device)
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    with args.output_scp.open("w") as scp, torch.inference_mode():
        for key, path in tqdm(args.entries, desc="EnCodec"):
            wav, sr = torchaudio.load(path)
            wav = torchaudio.functional.resample(wav.mean(0), sr, processor.sampling_rate)
            inputs = processor(raw_audio=wav.numpy(), sampling_rate=processor.sampling_rate, return_tensors="pt")
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            encoded = model.encode(inputs["input_values"], inputs["padding_mask"], bandwidth=12.0)
            audio = model.decode(encoded.audio_codes, encoded.audio_scales, inputs["padding_mask"])[0]
            key, target = output_path(args, key, "EnCodec")
            save_stimulus(target, audio.reshape(-1).cpu().numpy(), processor.sampling_rate)
            scp.write(f"{key} {target.as_posix()}\n")


if __name__ == "__main__":
    main()
