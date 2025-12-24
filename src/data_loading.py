"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio

# Ces fonctions existent dans ton projet (d'après tes extraits).
# Il y a une typo possible "preporcessing". On gère les deux.
try:
    from .preprocessing import get_preprocess_transforms
except Exception:
    from .preporcessing import get_preprocess_transforms  # type: ignore

from .augmentation import get_augmentation_transforms


def _speechcommands_labels() -> List[str]:
    # Labels standards SpeechCommands (35 classes).
    return [
        "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
        "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
        "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
        "tree", "two", "up", "visual", "wow", "yes", "zero",
    ]


class SpeechCommandsWrapper(Dataset):
    """
    Dataset SpeechCommands (torchaudio) avec preprocess + augment (train).
    Chaque item retourne (x, y) où:
      - x : Tensor audio (forme typique [1, T] ou autre selon preprocess)
      - y : int (classe)
    """

    def __init__(
        self,
        root: str,
        subset: str,
        download: bool,
        preprocess=None,
        augment=None,
        labels: Optional[List[str]] = None,
    ):
        assert subset in {"training", "validation", "testing"}
        self.ds = torchaudio.datasets.SPEECHCOMMANDS(
            root=root,
            subset=subset,
            download=download,
        )
        self.preprocess = preprocess
        self.augment = augment

        self.labels = labels if labels is not None else _speechcommands_labels()
        self.label_to_idx = {lab: i for i, lab in enumerate(self.labels)}

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        waveform, sample_rate, label, *_ = self.ds[idx]  # waveform: [1, T]
        if label not in self.label_to_idx:
            raise ValueError(f"Label inconnu reçu: {label}")

        x = waveform
        if self.preprocess is not None:
            # On accepte (x, sample_rate=...) au cas où ton preprocess fait du resample.
            try:
                x = self.preprocess(x, sample_rate=sample_rate)
            except TypeError:
                x = self.preprocess(x)

        if self.augment is not None:
            x = self.augment(x)

        y = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return x, y


def _pad_or_truncate(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Pad/truncate sur la dimension temps.
    x attendu en [C, T].
    """
    if x.dim() != 2:
        raise ValueError(f"Attendu [C, T], reçu {tuple(x.shape)}")
    c, t = x.shape
    if t > target_len:
        return x[:, :target_len]
    if t < target_len:
        return torch.nn.functional.pad(x, (0, target_len - t))
    return x


def make_collate_fn(target_num_samples: Optional[int] = None):
    """
    Collate function pour batch audio à longueur variable.
    - si target_num_samples est fourni : pad/truncate à cette taille
    - sinon : pad à la taille max dans le batch
    """
    def collate(batch):
        xs, ys = zip(*batch)

        # Assure que chaque x est [C, T]
        xs = [x if x.dim() == 2 else x.unsqueeze(0) for x in xs]

        if target_num_samples is None:
            max_len = max(x.shape[1] for x in xs)
        else:
            max_len = int(target_num_samples)

        xs = torch.stack([_pad_or_truncate(x, max_len) for x in xs], dim=0)  # [B, C, T]
        ys = torch.stack(list(ys), dim=0)  # [B]
        return xs, ys

    return collate


def get_dataloaders(config: dict):
    """
    Crée et retourne les DataLoaders d'entraînement/validation/test et des métadonnées.
    """
    # -------- lecture config --------
    ds_cfg = config.get("dataset", {})
    train_cfg = config.get("train", {})
    preprocess_cfg = config.get("preprocess", {})

    ds_name = ds_cfg.get("name", "SPEECHCOMMANDS")
    if str(ds_name).upper() not in {"SPEECHCOMMANDS", "SPEECH_COMMANDS", "SPEECH-COMMANDS"}:
        # Tu peux enlever ce check si ton projet gère d'autres datasets.
        raise ValueError(
            f"dataset.name='{ds_name}' non supporté par cette implémentation. "
            "Mets 'SPEECHCOMMANDS' dans le yaml."
        )

    root = ds_cfg.get("root", "./data")
    download = bool(ds_cfg.get("download", True))
    num_workers = int(ds_cfg.get("num_workers", 4))
    shuffle = bool(ds_cfg.get("shuffle", True))

    batch_size = int(train_cfg.get("batch_size", 64))

    # Optionnel : taille fixe pour input_shape stable.
    # Ton YAML n'a pas ce champ, donc on propose une valeur raisonnable si tu veux.
    # Tu peux ajouter: preprocess: { target_num_samples: 16000 }
    target_num_samples = preprocess_cfg.get("target_num_samples", None)
    if target_num_samples is not None:
        target_num_samples = int(target_num_samples)

    os.makedirs(root, exist_ok=True)

    # -------- transforms --------
    preprocess = get_preprocess_transforms(config)  # callable ou None
    augment_train = get_augmentation_transforms(config)  # callable ou None (train only)

    labels = _speechcommands_labels()

    # -------- datasets (TOUT le dataset via splits officiels) --------
    train_ds = SpeechCommandsWrapper(
        root=root,
        subset="training",
        download=download,
        preprocess=preprocess,
        augment=augment_train,
        labels=labels,
    )
    val_ds = SpeechCommandsWrapper(
        root=root,
        subset="validation",
        download=download,
        preprocess=preprocess,
        augment=None,
        labels=labels,
    )
    test_ds = SpeechCommandsWrapper(
        root=root,
        subset="testing",
        download=download,
        preprocess=preprocess,
        augment=None,
        labels=labels,
    )

    collate_fn = make_collate_fn(target_num_samples=target_num_samples)

    # -------- dataloaders --------
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # -------- meta --------
    num_classes = len(labels)

    # input_shape demandé : tuple
    # Audio mono => C=1, T variable => -1, ou fixe si target_num_samples est défini.
    if target_num_samples is None:
        input_shape = (1, -1)
    else:
        input_shape = (1, int(target_num_samples))

    meta: Dict = {
        "num_classes": num_classes,
        "input_shape": input_shape,
        "labels": labels,
    }

    return train_loader, val_loader, test_loader, meta

