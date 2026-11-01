"""Trains and evaluates the Alzheimer MRI classifier, then writes results/.

Data: the two parquet files from https://huggingface.co/datasets/Falah/Alzheimer_MRI
Usage: python train.py --data-dir data --seeds 0 1 2

Accuracy alone misleads here: Moderate Demented is under 1% of the data, so this reports
balanced accuracy and macro F1 too, each as a mean over seeds on the untouched test split.
"""
import argparse
import io
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ["Mild", "Moderate", "Non", "Very mild"]
TRAIN = "train-00000-of-00001-c08a401c53fe5312.parquet"
TEST = "test-00000-of-00001-44110b9df98c5585.parquet"


def load(path):
    frame = pd.read_parquet(path)
    images = np.stack([np.asarray(Image.open(io.BytesIO(i["bytes"])).convert("L")) for i in frame["image"]])
    return images[:, None].astype(np.float32) / 255.0, frame["label"].to_numpy()


class CNN(nn.Module):
    """Two convolution blocks and a flatten head.

    Chosen on validation macro F1. Deeper variants with batch normalization and global
    pooling scored far lower on this dataset (0.31 against 0.93 after 15 epochs).
    """

    def __init__(self, classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.2),
        )
        self.head = nn.Linear(64 * 32 * 32, classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


@torch.no_grad()
def predict(model, x, device, batch=128):
    model.eval()
    return np.concatenate([model(torch.from_numpy(x[i:i + batch]).to(device)).argmax(1).cpu().numpy()
                           for i in range(0, len(x), batch)])


def scores(y, p):
    return dict(accuracy=accuracy_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p),
                macro_f1=f1_score(y, p, average="macro"))


def train_one(x, y, x_test, y_test, seed, device, epochs, patience):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Stratified, so all 49 Moderate scans are not left to chance as in an unshuffled split.
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.15, stratify=y, random_state=seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr).long()),
                        batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = CNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.5, patience=4)
    best, best_state, wait, log = -1.0, None, 0, []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        val = f1_score(y_val, predict(model, x_val, device), average="macro")
        scheduler.step(val)
        log.append(dict(epoch=epoch, train_loss=total / len(x_tr), val_macro_f1=val))
        print(f"seed {seed} epoch {epoch} loss {total / len(x_tr):.4f} val macro F1 {val:.4f}", flush=True)
        # Model selection uses validation only. The test split is touched once, below.
        if val > best:
            best, wait = val, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, predict(model, x_test, device), log


def grad_cam(model, x, device):
    model.eval()
    grabbed = {}
    last = model.features[4]  # second convolution

    def keep_activation(_module, _inputs, output):
        output.retain_grad()
        grabbed["a"] = output

    handle = last.register_forward_hook(keep_activation)
    xt = torch.from_numpy(x).to(device)
    logits = model(xt)
    logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
    handle.remove()
    a = grabbed["a"]
    cam = F.relu((a.grad.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, x.shape[2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam.detach().cpu().numpy(), logits.argmax(1).cpu().numpy()


def near_duplicate_rate(x, x_test, threshold=0.98):
    """Share of test images almost identical to a training image, a leakage signal.

    The dataset has no patient IDs, so neighbouring slices of one patient can sit in both splits.
    """
    def small(a):
        t = F.avg_pool2d(torch.from_numpy(a), 4).flatten(1)
        return F.normalize(t - t.mean(1, keepdim=True), dim=1)
    best = (small(x_test) @ small(x).T).max(dim=1).values.numpy()
    return dict(threshold=threshold, share_above=float(np.mean(best > threshold)),
                median_nearest_correlation=float(np.median(best)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    data = pathlib.Path(args.data_dir)
    x, y = load(data / TRAIN)
    x_test, y_test = load(data / TEST)

    flat, flat_test = x.reshape(len(x), -1), x_test.reshape(len(x_test), -1)
    baselines = {
        "Always predict the largest class": scores(y_test, DummyClassifier().fit(flat, y).predict(flat_test)),
        "Logistic regression on pixels": scores(y_test, LogisticRegression(max_iter=300, class_weight="balanced")
                                                .fit(flat, y).predict(flat_test)),
    }
    runs, logs, best_macro, best_model, best_pred = [], {}, -1, None, None
    for seed in args.seeds:
        model, pred, log = train_one(x, y, x_test, y_test, seed, device, args.epochs, args.patience)
        run = scores(y_test, pred)
        run.update(seed=seed, epochs_run=len(log))
        runs.append(run)
        logs[seed] = log
        print("TEST", run, flush=True)
        # Which seed gets plotted is cosmetic. Reported numbers are the mean over all seeds.
        if seed == args.seeds[0]:
            best_model, best_pred = model, pred

    summary = {m: dict(mean=float(np.mean([r[m] for r in runs])), std=float(np.std([r[m] for r in runs])))
               for m in ("accuracy", "balanced_accuracy", "macro_f1")}
    report = classification_report(y_test, best_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    result = dict(baselines=baselines, cnn_summary=summary, cnn_runs=runs, leakage=near_duplicate_rate(x, x_test),
                  train_counts=np.bincount(y).tolist(), test_counts=np.bincount(y_test).tolist(),
                  first_seed_report=report, device=device)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), CLASSES)
    ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] / cm[i].sum() > .5 else "black")
    ax.set_title(f"Test confusion matrix, seed {args.seeds[0]}")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png")

    pick = [np.flatnonzero(y_test == k)[0] for k in range(4)]
    cam, pred = grad_cam(best_model, x_test[pick], device)
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9), dpi=160)
    for ax, i, c, p, k in zip(axes, pick, cam, pred, range(4)):
        ax.imshow(x_test[i, 0], cmap="gray")
        ax.imshow(c, cmap="inferno", alpha=0.4)
        ax.set_title(f"True {CLASSES[k]}, predicted {CLASSES[p]}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "grad_cam.png")
    print("SUMMARY", json.dumps({k: result[k] for k in ("baselines", "cnn_summary", "leakage")}, indent=2))


if __name__ == "__main__":
    main()
