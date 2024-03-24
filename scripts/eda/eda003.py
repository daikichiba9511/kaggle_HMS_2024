import torch
import torchaudio

spec_module = torchaudio.transforms.MelSpectrogram(
    sample_rate=200,
    n_fft=1024,
    hop_length=7500 // 256,
    win_length=128,
    n_mels=128,
    f_min=0,
    f_max=20,
    power=2.0,
    pad_mode="constant",
    center=True,
)

waveform = torch.randn(4, 10000)
spec = spec_module(waveform)
w, h = (spec.shape[1] // 32) * 32, (spec.shape[2] // 32) * 32
spec = spec[:, :w, :h]
print(f"I: {waveform.shape}, O: {spec.shape}")

waveform = torch.randn(4, 7500)
spec = spec_module(waveform)
w, h = (spec.shape[1] // 32) * 32, (spec.shape[2] // 32) * 32
spec = spec[:, :w, :h]
print(f"I: {waveform.shape}, O: {spec.shape}")
