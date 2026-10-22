"""Script de inferência/predição pronto para submissão no Kaggle.

Uso:
    python predict.py --checkpoint /kaggle/working/outputs/model_final.pth \
                      --output /kaggle/working/submission.csv
"""

import argparse
import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import DummyModel, load_config, set_seed


def dummy_test_loader(batch_size=32, n_samples=200, input_dim=10):
    """Dados de teste sintéticos. Substitua pelo seu test set real."""
    X = torch.randn(n_samples, input_dim)
    ids = [f"test_{i}" for i in range(n_samples)]
    dataset = torch.utils.data.TensorDataset(X)
    return ids, DataLoader(dataset, batch_size=batch_size, shuffle=False)


def predict(cfg, checkpoint_path, output_path):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    model = DummyModel().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    ids, test_loader = dummy_test_loader(batch_size=cfg["batch_size"])

    predictions = []
    with torch.no_grad():
        for (inputs,) in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(probs.flatten())

    submission = pd.DataFrame({"id": ids, "target": predictions})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submissão salva: {output_path} ({len(submission)} linhas)")
    print(submission.head())


def main():
    parser = argparse.ArgumentParser(description="Inferência no Kaggle")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="/kaggle/working/submission.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    predict(cfg, args.checkpoint, args.output)


if __name__ == "__main__":
    main()