from __future__ import annotations

from typing import Callable, Optional, Dict, Any

import torchaudio


def get_augmentation_transforms(config: dict) -> Optional[Callable]:
    """
    Retourne un callable:
      augment(spec: Tensor) -> Tensor

    Ici on applique SpecAugment sur les log-mel spectrogrammes:
    - FrequencyMasking
    - TimeMasking

    Le spec attendu est [1, F, T].
    """
    aug: Dict[str, Any] = config.get("augment", {}) or {}
    spec_cfg = aug.get("spec_augment", None)

    if spec_cfg in (None, False):
        return None

    # Si spec_augment: true => valeurs par défaut raisonnables
    if spec_cfg is True:
        freq_mask_param = 8
        time_mask_param = 20
        num_freq_masks = 1
        num_time_masks = 1
    elif isinstance(spec_cfg, dict):
        freq_mask_param = int(spec_cfg.get("freq_mask_param", 8))
        time_mask_param = int(spec_cfg.get("time_mask_param", 20))
        num_freq_masks = int(spec_cfg.get("num_freq_masks", 1))
        num_time_masks = int(spec_cfg.get("num_time_masks", 1))
    else:
        return None

    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

    def augment(spec):
        x = spec
        for _ in range(num_freq_masks):
            x = freq_mask(x)
        for _ in range(num_time_masks):
            x = time_mask(x)
        return x

    return augment

