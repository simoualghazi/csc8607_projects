from src.data_loading import get_dataloaders
import yaml
import torch

# Charger le YAML
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

train_loader, val_loader, test_loader, meta = get_dataloaders(config)

print("=== META ===")
for k, v in meta.items():
    print(f"{k}: {v}")

print("\n=== DATASET SIZES ===")
print("Train:", len(train_loader.dataset))
print("Val  :", len(val_loader.dataset))
print("Test :", len(test_loader.dataset))

# Prendre un batch
x, y = next(iter(train_loader))

print("\n=== BATCH INFO ===")
print("x.shape:", x.shape)
print("y.shape:", y.shape)
print("y[:10]:", y[:10])

print("\nCUDA available:", torch.cuda.is_available())

