"""
Audio preprocessing transforms.
"""

import torch
import torchaudio.transforms as T


class LogMelSpectrogram:
    """Compute log-Mel spectrograms from raw waveforms.

    Attributes:
        mel (T.MelSpectrogram): convert waveform→Mel spectrogram.
        db (T.AmplitudeToDB): convert power→decibel scale.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
    ):
        """Initialize the Mel and dB transforms.

        Args:
            sample_rate (int): Audio sampling rate in Hz.
            n_fft (int): FFT window size.
            hop_length (int): Hop length between STFT frames.
            n_mels (int): Number of Mel filterbanks.
        """
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.db = T.AmplitudeToDB(stype="power")

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply transforms to waveform.

        Args:
            waveform (Tensor [1, N]): Single-channel audio tensor.

        Returns:
            Tensor [n_mels, T]: Log-Mel spectrogram.
        """
        mel_spec = self.mel(waveform)  # [1, n_mels, T]
        log_mel = self.db(mel_spec)  # [1, n_mels, T]
        return log_mel.squeeze(0)  # [n_mels, T]


class SpecAugment:
    """
    Apply on-the-fly SpecAugment:
    - Random frequency masking
    - Random time masking
    """

    def __init__(self, freq_mask_param: int = 15, time_mask_param: int = 50):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: [1, n_mels, T] or [n_mels, T]
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
            _, n_mels, T = spec.shape
            f = torch.randint(0, self.freq_mask_param, (1,)).item()
            t = torch.randint(0, self.time_mask_param, (1,)).item()
            spec[:, :f, :] = 0
            spec[:, :, :t] = 0
        return spec.squeeze(0)
