import os
import yaml
import torch
import torch.nn as nn

from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed, get_device


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--n", type=int, default=128, help="Taille du sous-ensemble train")
    p.add_argument("--epochs", type=int, default=30, help="Nombre d'époques")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--wd", type=float, default=0.0, help="Weight decay (0 recommandé)")
    args = p.parse_args()

    # Charger config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Seed + device
    seed = int(config["train"].get("seed", 42))
    set_seed(seed)
    device = get_device(config["train"].get("device", "auto"))

    # Data loaders "normaux"
    train_loader, _, _, meta = get_dataloaders(config)

    # Sous-ensemble (N premiers exemples)
    n = min(args.n, len(train_loader.dataset))
    subset_ds = Subset(train_loader.dataset, list(range(n)))

    # IMPORTANT: réutiliser le même collate_fn que le train_loader
    small_loader = DataLoader(
        subset_ds,
        batch_size=min(int(config["train"].get("batch_size", 64)), n),
        shuffle=True,
        num_workers=0,                 # plus simple/debug
        collate_fn=train_loader.collate_fn
    )

    # Modèle
    model = build_model(config).to(device)

    # Loss + Optim
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # TensorBoard
    runs_dir = config["paths"].get("runs_dir", "./runs")
    os.makedirs(runs_dir, exist_ok=True)

    variant = config["model"].get("channels_variant", "small")
    n_fft = config["preprocess"].get("n_fft", 400)
    run_name = f"overfit_N={n}_variant={variant}_nfft={n_fft}_lr={args.lr}_wd={args.wd}"
    writer = SummaryWriter(log_dir=os.path.join(runs_dir, run_name))

    print("=== OVERFIT SMALL ===")
    print("N:", n)
    print("channels_variant:", variant)
    print("n_fft:", n_fft)
    print("lr:", args.lr, "wd:", args.wd)
    print("input_shape:", meta["input_shape"], "num_classes:", meta["num_classes"])

    # Entraînement
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0

        for x, y in small_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            loss_sum += loss.item() * bs
            n_seen += bs

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            global_step += 1

        train_loss = loss_sum / n_seen
        writer.add_scalar("train/loss", train_loss, epoch)

        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.6f}")

    # Sauvegarde checkpoint
    artifacts_dir = config["paths"].get("artifacts_dir", "./artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    ckpt_path = os.path.join(artifacts_dir, "overfit_small.ckpt")
    torch.save({"model_state": model.state_dict(), "config": config, "meta": meta}, ckpt_path)
    print("Saved:", ckpt_path)

    writer.close()
    print("Done. Ouvre TensorBoard pour la courbe train/loss.")


if __name__ == "__main__":
    main()

