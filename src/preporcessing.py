from __future__ import annotations

from typing import Callable, Optional, Dict, Any

import torch
import torchaudio


def get_preprocess_transforms(config: dict) -> Optional[Callable]:
    """
    Retourne un callable:
      preprocess(waveform: Tensor, sample_rate: int) -> Tensor

    Pour SpeechCommands:
    - resample vers 16kHz
    - pad/truncate à 1 seconde (16000 samples) pour avoir T constant
    - MelSpectrogram -> AmplitudeToDB (log-mel)
    - retourne un tenseur 1 x F x T (canal=1)
    - normalisation optionnelle
    """
    pp: Dict[str, Any] = config.get("preprocess", {}) or {}

    target_sr = int(pp.get("target_sample_rate", 16000))
    target_num_samples = pp.get("target_num_samples", 16000)
    target_num_samples = int(target_num_samples) if target_num_samples is not None else 16000

    # Hyperparamètre à régler (exigé par l'énoncé)
    # Choix suggérés: 400 / 512 / 640
    n_fft = int(pp.get("n_fft", 400))
    win_length = pp.get("win_length", None)
    win_length = int(win_length) if win_length is not None else n_fft
    hop_length = pp.get("hop_length", None)
    hop_length = int(hop_length) if hop_length is not None else n_fft // 2

    n_mels = int(pp.get("n_mels", 64))

    # Normalisation simple (optionnelle)
    # ex yaml:
    # preprocess:
    #   spec_normalize: "standard"  # "standard" | "none"
    spec_norm = str(pp.get("spec_normalize", "standard")).lower()

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=2.0,
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def preprocess(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        # waveform: [1, T]
        x = waveform

        # 1) Resample si besoin
        if sample_rate != target_sr:
            x = torchaudio.functional.resample(x, sample_rate, target_sr)

        # 2) Fixer à 1 seconde (pad/truncate)
        t = x.shape[-1]
        if t > target_num_samples:
            x = x[..., :target_num_samples]
        elif t < target_num_samples:
            x = torch.nn.functional.pad(x, (0, target_num_samples - t))

        # 3) MelSpectrogram -> log-mel (dB)
        # mel: [1, n_mels, time]
        spec = mel(x)
        spec = to_db(spec)

        # 4) Normalisation (sur le spectrogramme)
        if spec_norm == "standard":
            mean = spec.mean()
            std = spec.std().clamp_min(1e-8)
            spec = (spec - mean) / std

        # 5) Forcer format entrée CNN: 1 x F x T
        # Ici spec est déjà [1, F, T]
        return spec

    return preprocess

