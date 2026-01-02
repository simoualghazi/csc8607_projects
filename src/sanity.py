import os
import yaml
import torch
import matplotlib.pyplot as plt

from src.data_loading import get_dataloaders


def save_spec_image(spec_1xfxT: torch.Tensor, title: str, path: str):
    """
    spec_1xfxT: Tensor [1, F, T] ou [F, T]
    Sauvegarde une image du spectrogramme.
    """
    if spec_1xfxT.dim() == 3:
        spec = spec_1xfxT[0]  # [F, T]
    else:
        spec = spec_1xfxT

    plt.figure(figsize=(6, 4))
    plt.imshow(spec.cpu().numpy(), aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Mel bins (F)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    # Charger config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, test_loader, meta = get_dataloaders(config)

    os.makedirs("artifacts/figures", exist_ok=True)

    # --------------------
    # D11: shape batch + cohérence meta
    # --------------------
    x_batch, y_batch = next(iter(train_loader))

    print("=== D11: BATCH SHAPES ===")
    print("x_batch.shape:", tuple(x_batch.shape))
    print("y_batch.shape:", tuple(y_batch.shape))

    batch_input_shape = tuple(x_batch.shape[1:])  # sans batch
    print("meta['input_shape']:", meta["input_shape"])
    print("batch input shape :", batch_input_shape)

    is_consistent = (tuple(meta["input_shape"]) == batch_input_shape)
    print("CONSISTENT meta vs batch:", is_consistent)

    # --------------------
    # D10: 2–3 exemples après preprocessing/augmentation
    # On prend directement des éléments du dataset train (inclut augmentations)
    # --------------------
    print("\n=== D10: Saving 3 examples (train dataset, after preprocess/augment) ===")
    for i in range(3):
        x_ex, y_ex = train_loader.dataset[i]  # x_ex: [1, F, T] (attendu)
        label_name = meta["labels"][int(y_ex)]

        out_path = f"artifacts/figures/sanity_example_{i}_label_{label_name}.png"
        save_spec_image(
            x_ex,
            title=f"Example {i} - label: {label_name} - shape: {tuple(x_ex.shape)}",
            path=out_path
        )
        print("saved:", out_path)

    # Sauvegarder aussi une image d'un exemple DU BATCH (pour montrer format batch)
    out_path_batch0 = "artifacts/figures/sanity_batch0.png"
    save_spec_image(
        x_batch[0],  # [1, F, T]
        title=f"Batch[0] - shape: {tuple(x_batch[0].shape)}",
        path=out_path_batch0
    )
    print("saved:", out_path_batch0)


if __name__ == "__main__":
    main()


