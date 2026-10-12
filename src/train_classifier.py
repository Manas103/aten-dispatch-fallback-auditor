"""Trains `TinyEncoder` on the synthetic sum-mod-10 task (task.py) to a
real, nonzero accuracy, then saves fp32 weights. This is the only
training this project does; everything downstream (export, NNCF
calibration, the accuracy gate, the throughput benchmark) operates on
these fixed, trained weights, never a randomly initialized model, so the
INT8 accuracy comparison means something.
"""
import time

import torch
import torch.nn.functional as F

from model import TinyEncoder
from task import N_CLASSES, make_dataset, iterate_batches

ARTIFACT_DIR = "artifacts"
TRAIN_SEED = 1234
VAL_SEED = 5678
N_TRAIN = 4096
N_VAL = 1024
EPOCHS = 12
LR = 3e-3


def evaluate(model, token_ids, labels) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in iterate_batches(token_ids, labels):
            logits = model(xb)
            pred = logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.shape[0]
    return correct / total


def main():
    import os
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    torch.manual_seed(TRAIN_SEED)
    model = TinyEncoder(n_classes=N_CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_ids, train_labels = make_dataset(N_TRAIN, TRAIN_SEED)
    val_ids, val_labels = make_dataset(N_VAL, VAL_SEED)

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in iterate_batches(train_ids, train_labels):
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        val_acc = evaluate(model, val_ids, val_labels)
        print(f"epoch {epoch:2d}  loss {epoch_loss / n_batches:.4f}  val_acc {val_acc:.4f}")
    elapsed = time.perf_counter() - t0

    final_acc = evaluate(model, val_ids, val_labels)
    print(f"trained in {elapsed:.1f}s, final val top-1 accuracy: {final_acc:.4f}")

    torch.save(model.state_dict(), f"{ARTIFACT_DIR}/model_fp32.pt")
    torch.save({"token_ids": val_ids, "labels": val_labels}, f"{ARTIFACT_DIR}/val_set.pt")
    print(f"saved {ARTIFACT_DIR}/model_fp32.pt and {ARTIFACT_DIR}/val_set.pt")


if __name__ == "__main__":
    main()
