import numpy as np
import torch
import torchaudio

from src.training import data as my_data
from src.training import preprocessings as my_preprocessings

spec_module = torchaudio.transforms.MelSpectrogram(
    sample_rate=200,
    n_fft=1024,
    hop_length=10000 // 256,
    win_length=128,
    n_mels=128,
    f_min=0,
    f_max=20,
    power=2.0,
    pad_mode="constant",
    center=True,
)


def make_specs_from_signals(
    signals: torch.Tensor,
    spec_module: torchaudio.transforms.Spectrogram | torchaudio.transforms.MelSpectrogram,
    db_transform: torchaudio.transforms.AmplitudeToDB | None = None,
    random_crop_signal_augmentation: bool = False,
    crop_signal_length: int = 7500,
) -> torch.Tensor:
    bs = signals.size(0)
    # signal to specs
    # signal shape: (bs, 4, 4, 10000)
    specs = torch.zeros((bs, 4, 128, 256))
    for k in range(4):
        for kk in range(4):
            # spec shape: (bs, 129, 257)
            signal = signals[:, k, kk, :]
            if random_crop_signal_augmentation:
                start_point = np.random.randint(0, 10000 - crop_signal_length)
                signal = signal[:, start_point : start_point + crop_signal_length]

            reduced_signals = my_preprocessings.butter_lowpass_filter(signals.numpy())
            reduced_signals = my_preprocessings.quantize_data(reduced_signals, classes=256)
            signals = torch.from_numpy(reduced_signals).float()

            spec = spec_module(signal)
            if db_transform is not None:
                spec = db_transform(spec)

            height = (spec.shape[1] // 32) * 32
            width = (spec.shape[2] // 32) * 32
            # shape: (bs, 128, 256)
            spec = spec[:, :height, :width]

            print(f"spec value range {spec.min()} ~ {spec.max()}")
            print(spec.shape)

            spec = (spec + 40) / 40
            specs[:, k, :, :] += spec

        # clip outliers for stability and standardization
        print(f"specs value range {specs[:, k, :, :].min()} ~ {specs[:, k, :, :].max()}")
        specs[:, k, :, :] /= 4.0
        specs[:, k, :, :] = torch.clamp(specs[:, k, :, :], min=np.exp(-4.0), max=np.exp(8.0))
        specs[:, k, :, :] = torch.log(specs[:, k, :, :]) / 4.0
        print(f"clamped & log transfromed specs value range {specs[:, k, :, :].min()} ~ {specs[:, k, :, :].max()}")

    return specs


db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
dl = my_data.init_valid_dataloader(0, batch_size=2, is_debug=True, num_workers=0)
for batch_idx, batch in enumerate(dl):
    signals = batch["signals"]
    specs = make_specs_from_signals(
        signals, spec_module, db_transform=db_transform, random_crop_signal_augmentation=False
    )
    print(f"signals: {signals.shape}, specs: {specs.shape}")
    break
