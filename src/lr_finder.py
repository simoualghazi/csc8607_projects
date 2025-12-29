import os
import yaml
import math
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed, get_device


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_lr", type=float, default=1.0)
    p.add_argument("--num_iters", type=int, default=100)
    args = p.parse_args()

    # Config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["train"]["seed"])
    device = get_device(config["train"]["device"])

    train_loader, _, _, meta = get_dataloaders(config)

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.min_lr)

    # TensorBoard
    runs_dir = config["paths"]["runs_dir"]
    os.makedirs(runs_dir, exist_ok=True)

    variant = config["model"]["channels_variant"]
    n_fft = config["preprocess"]["n_fft"]
    run_name = f"lr_finder_variant={variant}_nfft={n_fft}"
    writer = SummaryWriter(log_dir=os.path.join(runs_dir, run_name))

    # LR schedule (log-scale)
    lrs = torch.logspace(
        math.log10(args.min_lr),
        math.log10(args.max_lr),
        steps=args.num_iters
    )

    print("=== LR FINDER ===")
    print("min_lr:", args.min_lr, "max_lr:", args.max_lr)

    data_iter = iter(train_loader)

    for step, lr in enumerate(lrs):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr.item()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        writer.add_scalar("lr", lr.item(), step)
        writer.add_scalar("loss", loss.item(), step)

        if step % 10 == 0:
            print(f"step {step:03d} | lr={lr:.2e} | loss={loss.item():.4f}")

        if not torch.isfinite(loss):
            print("Loss diverged, stopping early.")
            break

    writer.close()
    print("LR finder terminé. Ouvre TensorBoard.")


if __name__ == "__main__":
    main()

