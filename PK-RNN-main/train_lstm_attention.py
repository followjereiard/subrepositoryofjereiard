import pathlib
from typing import Iterable, Tuple

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import Example, collate_batch
from model import LSTMAttentionVancomycin

TensorTuple = Tuple[torch.Tensor, ...]


def _to_device(batch: Tuple, device: torch.device) -> Tuple:
    moved = []
    for item in batch:
        if isinstance(item, torch.Tensor):
            moved.append(item.to(device))
        else:
            moved.append(item)
    return tuple(moved)


def _train_epoch(
    model: LSTMAttentionVancomycin,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        optimizer.zero_grad()
        batch = _to_device(batch, device)
        _, loss = model(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def _evaluate(
    model: LSTMAttentionVancomycin,
    loader: Iterable,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            _, loss = model(batch)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / max(num_batches, 1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAttentionVancomycin(device=device).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(Example(), batch_size=8, collate_fn=collate_batch)
    val_loader = DataLoader(Example(), batch_size=8, collate_fn=collate_batch)
    epochs = 100
    patience = 10
    best_val = float("inf")
    wait = 0
    model_path = pathlib.Path("best_lstm_attention.pt")
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, device)
        val_loss = _evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        if val_loss < best_val:
            torch.save(model.state_dict(), model_path)
            best_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
