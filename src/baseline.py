"""
Baselines pour le projet SpeechCommands.

Calcule les performances de références :
1. Classe majoritaire : prédit toujours la classe la plus fréquente
2. Prédiction aléatoire uniforme : prédit uniformément parmi les 35 classes
3. Prédiction aléatoire pondérée : prédit selon la distribution des classes

Ces baselines servent de référence pour évaluer les modèles entraînés.
"""

import os
from collections import Counter

import torch
import yaml

from src.data_loading import get_dataloaders


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Calcule l'accuracy entre prédictions et labels."""
    return (preds == labels).float().mean().item()


def compute_class_distribution(loader) -> Counter:
    """Compte la distribution des classes dans un DataLoader."""
    counter = Counter()
    for _, labels in loader:
        counter.update(labels.tolist())
    return counter


def baseline_majority_class(loader, majority_class: int) -> float:
    """
    Baseline classe majoritaire.
    Prédit toujours la classe la plus fréquente.
    """
    total_acc = 0.0
    num_batches = 0
    
    for _, labels in loader:
        preds = torch.full_like(labels, fill_value=majority_class)
        total_acc += accuracy(preds, labels)
        num_batches += 1
    
    return total_acc / num_batches if num_batches > 0 else 0.0


def baseline_random_uniform(loader, num_classes: int) -> float:
    """
    Baseline prédiction aléatoire uniforme.
    Prédit uniformément parmi les classes.
    """
    total_acc = 0.0
    num_batches = 0
    
    for _, labels in loader:
        preds = torch.randint(low=0, high=num_classes, size=labels.shape)
        total_acc += accuracy(preds, labels)
        num_batches += 1
    
    return total_acc / num_batches if num_batches > 0 else 0.0


def baseline_random_weighted(loader, class_probs: torch.Tensor) -> float:
    """
    Baseline prédiction aléatoire pondérée.
    Prédit selon la distribution des classes observée dans l'entraînement.
    """
    total_acc = 0.0
    num_batches = 0
    
    for _, labels in loader:
        # Échantillonne selon les probabilités des classes
        preds = torch.multinomial(class_probs, num_samples=len(labels), replacement=True)
        total_acc += accuracy(preds, labels)
        num_batches += 1
    
    return total_acc / num_batches if num_batches > 0 else 0.0


def main():
    # Charger la configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Charger les DataLoaders
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    
    num_classes = meta["num_classes"]
    labels = meta["labels"]
    
    print(f"Dataset : SpeechCommands ({num_classes} classes)")
    print(f"Classes : {labels[:5]} ... (et {num_classes - 5} autres)")
    print("-" * 60)
    
    # Calculer la distribution des classes sur l'entraînement
    print("Analyse de la distribution des classes (train)...")
    train_distribution = compute_class_distribution(train_loader)
    
    # Trouver la classe majoritaire
    majority_class, majority_count = train_distribution.most_common(1)[0]
    total_samples = sum(train_distribution.values())
    
    print(f"Classe majoritaire : '{labels[majority_class]}' (index {majority_class})")
    print(f"  -> {majority_count}/{total_samples} échantillons ({100*majority_count/total_samples:.2f}%)")
    print("-" * 60)
    
    # Calculer les probabilités des classes pour le baseline pondéré
    class_counts = torch.tensor([train_distribution[i] for i in range(num_classes)], dtype=torch.float)
    class_probs = class_counts / class_counts.sum()
    
    # Évaluer sur le set de validation
    print("\n=== Évaluation sur le set de VALIDATION ===")
    
    acc_majority = baseline_majority_class(val_loader, majority_class)
    acc_random_uniform = baseline_random_uniform(val_loader, num_classes)
    acc_random_weighted = baseline_random_weighted(val_loader, class_probs)
    
    print(f"Baseline classe majoritaire    : {100*acc_majority:.2f}%")
    print(f"Baseline aléatoire uniforme    : {100*acc_random_uniform:.2f}%")
    print(f"Baseline aléatoire pondéré     : {100*acc_random_weighted:.2f}%")
    
    # Évaluer sur le set de test
    print("\n=== Évaluation sur le set de TEST ===")
    
    acc_majority_test = baseline_majority_class(test_loader, majority_class)
    acc_random_uniform_test = baseline_random_uniform(test_loader, num_classes)
    acc_random_weighted_test = baseline_random_weighted(test_loader, class_probs)
    
    print(f"Baseline classe majoritaire    : {100*acc_majority_test:.2f}%")
    print(f"Baseline aléatoire uniforme    : {100*acc_random_uniform_test:.2f}%")
    print(f"Baseline aléatoire pondéré     : {100*acc_random_weighted_test:.2f}%")
    
    
    # Sauvegarder les résultats
    os.makedirs("artifacts", exist_ok=True)
    results = {
        "num_classes": num_classes,
        "majority_class": {
            "index": int(majority_class),
            "name": labels[majority_class],
            "count": int(majority_count),
            "proportion": float(majority_count / total_samples),
        },
        "validation": {
            "majority_class": float(acc_majority),
            "random_uniform": float(acc_random_uniform),
            "random_weighted": float(acc_random_weighted),
        },
        "test": {
            "majority_class": float(acc_majority_test),
            "random_uniform": float(acc_random_uniform_test),
            "random_weighted": float(acc_random_weighted_test),
        },
        "theoretical": {
            "random_uniform": 1.0 / num_classes,
            "random_weighted": float(expected_weighted),
        },
    }
    
    with open("artifacts/baseline_results.yaml", "w") as f:
        yaml.dump(results, f, default_flow_style=False)


if __name__ == "__main__":
    main()

