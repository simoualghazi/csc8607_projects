"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

from __future__ import annotations
import torch
import torch.nn as nn


class CNN2DSpeech(nn.Module):
    """
    CNN 2D pour Speech Commands sur spectrogrammes log-mel.

    Entrée:  (B, 1, F, T)
    Bloc A:  Conv2d(1 -> C1, 3x3, pad=1) -> BN -> ReLU -> MaxPool2d(2)
    Bloc B:  Conv2d(C1 -> C2, 3x3, pad=1) -> BN -> ReLU -> MaxPool2d(2)
    Bloc C:  Conv2d(C2 -> C3, 3x3, pad=1) -> BN -> ReLU -> GlobalAvgPool
    Tête:    Linear(C3 -> num_classes)

    Sortie: (B, num_classes) = logits
    """

    def __init__(self, num_classes: int, channels_variant: str = "small", dropout: float = 0.0):
        super().__init__()

        if channels_variant == "small":
            c1, c2, c3 = 32, 64, 128
        elif channels_variant == "large":
            c1, c2, c3 = 48, 96, 192
        else:
            raise ValueError("channels_variant must be 'small' or 'large'.")

        self.block_a = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.block_b = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.block_c = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.gap(x)          # (B, C3, 1, 1)
        x = x.flatten(1)         # (B, C3)
        x = self.dropout(x)
        logits = self.fc(x)      # (B, num_classes)
        return logits


def build_model(config: dict) -> nn.Module:
    """
    Construit et retourne un nn.Module selon la config.

    Attendu dans config.yaml (recommandé) :
    model:
      type: "cnn2d"
      num_classes: 35
      channels_variant: "small"   # "small" ou "large"
      dropout: 0.0
    """
    mcfg = config.get("model", {}) or {}

    model_type = str(mcfg.get("type", "cnn2d")).lower()

    #  num_classes doit être connu. Mets-le dans le YAML: model.num_classes: 35
    num_classes = mcfg.get("num_classes", None)
    if num_classes is None:
        raise ValueError(
            "config['model']['num_classes'] est manquant. "
            "Ajoute dans configs/config.yaml:\n"
            "model:\n  num_classes: 35"
        )
    num_classes = int(num_classes)

    channels_variant = str(mcfg.get("channels_variant", "small")).lower()
    dropout = float(mcfg.get("dropout", 0.0))

    if model_type in {"cnn2d", "cnn2d_speech", "cnn"}:
        return CNN2DSpeech(num_classes=num_classes, channels_variant=channels_variant, dropout=dropout)

    raise ValueError(f"Type de modèle non supporté: {model_type}")
