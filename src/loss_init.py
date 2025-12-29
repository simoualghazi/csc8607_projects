import yaml
import torch
import torch.nn as nn

from src.data_loading import get_dataloaders
from src.model import build_model


def main():
    # Charger config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # Data
    train_loader, _, _, meta = get_dataloaders(config)

    # Modèle (UN SEUL argument)
    model = build_model(config)

    # Premier batch
    x, y = next(iter(train_loader))  # x: (B,1,64,81), y:(B,)

    # Loss
    criterion = nn.CrossEntropyLoss()

    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    print("batch shape :", tuple(x.shape))
    print("logits shape:", tuple(logits.shape))
    print("loss        :", float(loss.item()))
    print("grad nonzero:", model.fc.weight.grad.abs().sum().item() > 0)


if __name__ == "__main__":
    main()

