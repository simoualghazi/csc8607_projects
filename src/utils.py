"""
Utils génériques.

Fonctions attendues (signatures imposées) :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""

from __future__ import annotations

import os
import random
import yaml
import torch
import numpy as np


def set_seed(seed: int) -> None:
    """
    Initialise les seeds pour la reproductibilité.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproductibilité (attention: peut ralentir un peu)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = "auto") -> str:
    """
    Retourne le device à utiliser.
    - prefer="cpu"  -> force CPU
    - prefer="cuda" -> force CUDA (si dispo)
    - prefer="auto" -> CUDA si dispo, sinon CPU
    """
    if prefer is None or prefer == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    prefer = prefer.lower()
    if prefer == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] CUDA demandé mais non disponible, utilisation CPU.")
            return "cpu"
        return "cuda"

    if prefer == "cpu":
        return "cpu"

    raise ValueError(f"Device non reconnu: {prefer}")


def count_parameters(model) -> int:
    """
    Retourne le nombre total de paramètres entraînables du modèle.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config_snapshot(config: dict, out_dir: str) -> None:
    """
    Sauvegarde une copie de la config YAML dans out_dir/config_snapshot.yaml
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "config_snapshot.yaml")

    with open(out_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"[INFO] Config snapshot sauvegardée dans: {out_path}")
