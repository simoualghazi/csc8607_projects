import os
import yaml
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed, get_device, save_config_snapshot


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
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    args = p.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(int(config["train"]["seed"]))
    device = get_device(config["train"].get("device", "auto"))

    # Data
    train_loader, val_loader, _, meta = get_dataloaders(config)

    # Model
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer (no scheduler)
    lr = float(config["train"]["optimizer"]["lr"])
    wd = float(config["train"]["optimizer"]["weight_decay"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Logging + artifacts
    runs_dir = config["paths"]["runs_dir"]
    artifacts_dir = config["paths"]["artifacts_dir"]
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    variant = config["model"]["channels_variant"]
    n_fft = config["preprocess"]["n_fft"]
    epochs = int(config["train"]["epochs"])
    batch_size = int(config["train"]["batch_size"])

    run_name = f"full_lr={lr}_wd={wd}_variant={variant}_nfft={n_fft}_bs={batch_size}_ep={epochs}"
    writer = SummaryWriter(log_dir=os.path.join(runs_dir, run_name))

    # Save config snapshot
    save_config_snapshot(config, artifacts_dir)

    # Training loop + best checkpoint
    best_val_acc = -1.0
    best_path = os.path.join(artifacts_dir, "best.ckpt")

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

        print(f"epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        # Save best checkpoint by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "meta": meta,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                best_path,
            )
            print(f"  -> saved best to {best_path} (val_acc={best_val_acc:.4f})")

    writer.close()
    print("\nTraining finished.")
    print("Best checkpoint:", best_path)
    print("Best val_acc:", best_val_acc)


if __name__ == "__main__":
    main()
