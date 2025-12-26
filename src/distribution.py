from collections import Counter
import matplotlib.pyplot as plt
import yaml
import os

from src.data_loading import get_dataloaders

# Charger la config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

train_loader, val_loader, test_loader, meta = get_dataloaders(config)

labels = meta["labels"]

# ---- Compter les classes dans le TRAIN ----
counter = Counter()
for _, y in train_loader:
    counter.update(y.tolist())

# ---- Créer dossier de sortie ----
os.makedirs("artifacts/figures", exist_ok=True)

# ---- Graphique ----
counts = [counter[i] for i in range(len(labels))]

plt.figure(figsize=(14, 5))
plt.bar(range(len(labels)), counts)
plt.xticks(range(len(labels)), labels, rotation=90)
plt.ylabel("Number of samples")
plt.title("Class distribution in Speech Commands (Train split)")
plt.tight_layout()

# ✅ SAUVEGARDE DE L'IMAGE
plt.savefig("artifacts/figures/class_distribution_train.png", dpi=300)

plt.close()

