"""Script de treinamento pronto para rodar no Kaggle.

Uso:
    python train.py --epochs 5 --batch-size 32 --lr 2e-5
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DummyModel(nn.Module):
    """Modelo simples de exemplo. Substitua pelo seu modelo real."""
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def dummy_dataloader(batch_size=32, n_samples=1000, input_dim=10):
    """Gera dados sintéticos para exemplo. Substitua pelo seu dataset real."""
    X = torch.randn(n_samples, input_dim)
    y = (X[:, :5].sum(dim=1, keepdim=True) > 0).float()
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(cfg):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    model = DummyModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scaler = torch.cuda.amp.GradScaler() if cfg["mixed_precision"] else None

    train_loader = dummy_dataloader(batch_size=cfg["batch_size"])

    os.makedirs(cfg["output_dir"], exist_ok=True)

    model.train()
    accumulation_steps = cfg.get("gradient_accumulation_steps", 1)

    for epoch in range(cfg["epochs"]):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']}")
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / accumulation_steps
                scaler.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets) / accumulation_steps
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            pbar.set_postfix(loss=f"{running_loss / (i + 1):.4f}")

        # checkpoint a cada epoch
        ckpt_path = os.path.join(cfg["output_dir"], f"model_epoch{epoch + 1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint salvo: {ckpt_path}")

    final_path = os.path.join(cfg["output_dir"], "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Modelo final salvo: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Treinamento no Kaggle")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["learning_rate"] = args.lr
    if args.device is not None:
        cfg["device"] = args.device

    train(cfg)


if __name__ == "__main__":
    main()