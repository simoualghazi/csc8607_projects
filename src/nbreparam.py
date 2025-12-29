import yaml
import torch

from src.utils import set_seed, get_device, count_parameters, save_config_snapshot
from src.model import build_model
from src.data_loading import get_dataloaders


def main():
    # 1) Charger la config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2) Seed + device
    set_seed(int(config["train"]["seed"]))
    device = get_device(config["train"].get("device", "auto"))

    # 3) Construire le modèle
    model = build_model(config).to(device)

    # 4) Compter paramètres
    print("Total parameters:", count_parameters(model))

    # 5) Vérifier shapes entrée/sortie (utile pour ton rapport)
    _, _, _, meta = get_dataloaders(config)
    x_dummy = torch.zeros(2, *meta["input_shape"]).to(device)  # batch=2
    logits = model(x_dummy)
    print("meta['input_shape']:", meta["input_shape"])
    print("dummy input shape :", tuple(x_dummy.shape))
    print("logits shape      :", tuple(logits.shape))  # (2, num_classes)




if __name__ == "__main__":
    main()

