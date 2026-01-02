"""
Évaluation finale sur le jeu de test .

Usage :
python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
"""

import argparse
import yaml
import torch
import torch.nn as nn

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # -------- config --------
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(config["train"].get("device", "auto"))

    # -------- data --------
    _, _, test_loader, meta = get_dataloaders(config)
    num_classes = meta["num_classes"]

    # -------- model --------
    model = build_model(config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    # Pour accuracy par classe
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)

    # -------- evaluation --------
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            for c in range(num_classes):
                mask = (y == c)
                correct_per_class[c] += (preds[mask] == c).sum().item()
                total_per_class[c] += mask.sum().item()

    test_loss = total_loss / total
    test_acc = correct / total

    print("\n=== TEST EVALUATION ===")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_acc:.4f}")

    print("\nAccuracy par classe:")
    for i, label in enumerate(meta["labels"]):
        if total_per_class[i] > 0:
            acc_c = correct_per_class[i] / total_per_class[i]
            print(f"  {label:>10s}: {acc_c:.4f}")


if __name__ == "__main__":
    main()
