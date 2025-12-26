from src.data_loading import get_dataloaders
import yaml
import torch

# Charger le YAML
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

train_loader, val_loader, test_loader, meta = get_dataloaders(config)
#train_loader, val_loader , meta = get_dataloaders(confi
print("=== META ===")
for k, v in meta.items():
    print(f"{k}: {v}")

print("\n=== DATASET SIZES ===")
print("Train:", len(train_loader.dataset))
print("Val  :", len(val_loader.dataset))
print("Test :", len(test_loader.dataset))

# ---- Inspecter un batch TRAIN ----
print("\n=== TRAIN BATCH SHAPE ===")
x_train, y_train = next(iter(train_loader))
print("x_train.shape:", x_train.shape)   # [B, 1, F, T]
print("y_train.shape:", y_train.shape)   # [B]
print("y_train[:10]:", y_train[:10])

# ---- Inspecter un batch VAL ----
print("\n=== VAL BATCH SHAPE ===")
x_val, y_val = next(iter(val_loader))
print("x_val.shape:", x_val.shape)
print("y_val.shape:", y_val.shape)

# ---- Inspecter un batch TEST ----
print("\n=== TEST BATCH SHAPE ===")
x_test, y_test = next(iter(test_loader))
print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)

print("\nCUDA available:", torch.cuda.is_available())

# ---- Inspecter UN exemple du dataset (sans batch) ----
print("\n=== SINGLE EXAMPLE (TRAIN DATASET) ===")
x_ex, y_ex = train_loader.dataset[0]

print("x_ex.shape:", x_ex.shape)
print("y_ex:", y_ex)
print("x_ex.dim():", x_ex.dim())

