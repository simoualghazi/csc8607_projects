import os
import csv
import copy
import yaml
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed, get_device


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


def train_one_run(config, lr, wd, channels_variant, n_fft, epochs, run_name):
    # Config modifiée pour ce run
    cfg = copy.deepcopy(config)
    cfg["model"]["channels_variant"] = channels_variant
    cfg["preprocess"]["n_fft"] = int(n_fft)

    # Seed + device
    set_seed(int(cfg["train"]["seed"]))
    device = get_device(cfg["train"].get("device", "auto"))

    # Data (sera cohérent avec n_fft via preprocessing)
    train_loader, val_loader, _, meta = get_dataloaders(cfg)

    # Model
    model = build_model(cfg).to(device)

    # Optim + loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # TensorBoard
    runs_dir = cfg["paths"]["runs_dir"]
    os.makedirs(runs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(runs_dir, run_name))

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_seen = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            n_seen += bs

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/lr", lr, global_step)
            global_step += 1

        train_loss = train_loss_sum / n_seen
        val_loss, val_acc = evaluate(model, val_loader, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)

        print(f"[{run_name}] epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    writer.close()
    return val_loss, val_acc


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--epochs", type=int, default=3, help="1–5 époques (grid search rapide)")
    args = p.parse_args()

    # Charger config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Grilles (tu peux modifier)
    lrs = [5e-4, 1e-3, 2e-3]
    wds = [1e-5, 1e-4]
    variants = ["small", "large"]              # Hyp-A
    n_ffts = [400, 512, 640]                  # Hyp-B

    results_path = os.path.join(config["paths"]["artifacts_dir"], "grid_results.csv")
    os.makedirs(config["paths"]["artifacts_dir"], exist_ok=True)

    rows = []
    run_id = 0

    for lr in lrs:
        for wd in wds:
            for v in variants:
                for n_fft in n_ffts:
                    run_id += 1
                    run_name = f"grid_{run_id:02d}_lr={lr}_wd={wd}_v={v}_nfft={n_fft}"
                    val_loss, val_acc = train_one_run(
                        config=config,
                        lr=lr,
                        wd=wd,
                        channels_variant=v,
                        n_fft=n_fft,
                        epochs=args.epochs,
                        run_name=run_name,
                    )
                    rows.append({
                        "run": run_name,
                        "lr": lr,
                        "wd": wd,
                        "channels_variant": v,
                        "n_fft": n_fft,
                        "val_accuracy": val_acc,
                        "val_loss": val_loss,
                    })

    # Sauver CSV
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Meilleur run (max accuracy, puis min loss)
    best = sorted(rows, key=lambda r: (-r["val_accuracy"], r["val_loss"]))[0]
    print("\n=== BEST RUN ===")
    for k, v in best.items():
        print(k, ":", v)

    print("\nSaved results to:", results_path)


if __name__ == "__main__":
    main()

