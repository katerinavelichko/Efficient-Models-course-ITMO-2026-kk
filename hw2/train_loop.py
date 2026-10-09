import time
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


PAPER_EPOCHS = (1, 90, 45, 45)
PAPER_LRS = (0.01, 0.1, 0.01, 0.001)
PAPER_TOTAL = sum(PAPER_EPOCHS)


def scaled_epoch_lrs(total: int) -> List[float]:
    if total <= 3:
        return [0.1] * total
    counts = [max(1, int(round(n / PAPER_TOTAL * total))) for n in PAPER_EPOCHS]
    while sum(counts) > total:
        shrinkable = [i for i, c in enumerate(counts) if c > 1]
        counts[shrinkable[-1]] -= 1
    while sum(counts) < total:
        counts[1] += 1
    lrs: List[float] = []
    for count, lr in zip(counts, PAPER_LRS):
        lrs.extend([lr] * count)
    return lrs


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.shape[0])
    model.train()
    if total == 0:
        return float("nan")
    return correct / total


def predict_uncertainty(model: nn.Module, loader: DataLoader, n: int, device: torch.device) -> dict:
    entropy = np.zeros(n, dtype=np.float64)
    least_conf = np.zeros(n, dtype=np.float64)
    seen = np.zeros(n, dtype=bool)
    correct = 0
    total = 0
    model.eval()
    with torch.inference_mode():
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            log_p = torch.log_softmax(logits, dim=1)
            p = log_p.exp()
            h = -(p * log_p).sum(dim=1)
            lc = 1.0 - p.max(dim=1).values
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.shape[0])
            ids = idx.detach().cpu().numpy()
            entropy[ids] = h.detach().cpu().numpy()
            least_conf[ids] = lc.detach().cpu().numpy()
            seen[ids] = True
    missing = int((~seen).sum())
    if missing:
        raise RuntimeError(f"uncertainty pass missed {missing} / {n} examples")
    return {
        "entropy": entropy,
        "least_confidence": least_conf,
        "accuracy": correct / total if total else float("nan"),
    }


def collect_features(
    model: nn.Module,
    dataset,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    loader = DataLoader(
        Subset(dataset, [int(i) for i in indices]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    chunks = []
    model.eval()
    with torch.inference_mode():
        for x, _, _ in loader:
            _, feat = model.forward_features(x.to(device, non_blocking=True))
            chunks.append(feat.detach().cpu())
    return torch.cat(chunks, dim=0).numpy()


def _update_forgetting(forgetting, ever_correct, prev, ids, correct):
    was_correct = prev[ids]
    forgetting[ids] += np.logical_and(was_correct, np.logical_not(correct)).astype(np.int32)
    ever_correct[ids] |= correct
    prev[ids] = correct


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    lr_schedule: Sequence[float],
    *,
    test_loader: Optional[DataLoader] = None,
    eval_every: int = 1,
    n_train: Optional[int] = None,
    track_forgetting: bool = False,
    on_snapshot: Optional[Callable] = None,
    snapshot_epoch: Optional[int] = None,
    log_prefix: str = "",
    log_every: int = 1,
) -> dict:
    if track_forgetting and not n_train:
        raise ValueError("n_train is required when track_forgetting is set")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr_schedule[0], momentum=0.9, weight_decay=5e-4
    )
    forgetting = ever_correct = prev = None
    if track_forgetting:
        forgetting = np.zeros(n_train, dtype=np.int32)
        ever_correct = np.zeros(n_train, dtype=bool)
        prev = np.zeros(n_train, dtype=bool)

    history = []
    snapshot = None
    model.train()
    t0 = time.perf_counter()
    for epoch, lr in enumerate(lr_schedule, start=1):
        for group in optimizer.param_groups:
            group["lr"] = lr
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        epoch_t0 = time.perf_counter()
        for x, y, idx in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            if track_forgetting:
                correct = (pred == y).detach().cpu().numpy()
                ids = idx.detach().cpu().numpy()
                _update_forgetting(forgetting, ever_correct, prev, ids, correct)
            batch_n = int(y.shape[0])
            running_loss += float(loss.detach().item()) * batch_n
            running_correct += int((pred == y).sum().item())
            running_seen += batch_n
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        last = epoch == len(lr_schedule)
        do_eval = test_loader is not None and (last or (eval_every > 0 and epoch % eval_every == 0))
        test_acc = _accuracy(model, test_loader, device) if do_eval else float("nan")
        row = {
            "epoch": epoch,
            "lr": lr,
            "loss": running_loss / max(running_seen, 1),
            "aug_acc": running_correct / max(running_seen, 1),
            "test_acc": test_acc,
            "epoch_seconds": time.perf_counter() - epoch_t0,
        }
        history.append(row)
        if log_prefix and (last or (log_every > 0 and epoch % log_every == 0)):
            test_txt = f"{100 * test_acc:.2f}%" if test_acc == test_acc else "skip"
            print(
                f"{log_prefix} epoch {epoch:3d}/{len(lr_schedule)} lr={lr:.4g} "
                f"loss={row['loss']:.3f} aug_acc={100 * row['aug_acc']:.1f}% "
                f"test={test_txt} ({row['epoch_seconds']:.1f}s)",
                flush=True,
            )
        if snapshot_epoch is not None and epoch == snapshot_epoch and snapshot is None:
            snapshot = {
                "forgetting": None if forgetting is None else forgetting.copy(),
                "ever_correct": None if ever_correct is None else ever_correct.copy(),
                "test_acc": test_acc if test_acc == test_acc else _accuracy(model, test_loader, device) if test_loader is not None else float("nan"),
            }
            if on_snapshot is not None:
                snapshot.update(on_snapshot(model))
            model.train()

    test_acc = history[-1]["test_acc"] if history else float("nan")
    if test_loader is not None and test_acc != test_acc:
        test_acc = _accuracy(model, test_loader, device)
    return {
        "model": model,
        "seconds": time.perf_counter() - t0,
        "history": history,
        "test_acc": test_acc,
        "forgetting": forgetting,
        "ever_correct": ever_correct,
        "snapshot": snapshot,
    }
