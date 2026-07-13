"""1D-CNN classifier (PyTorch, CPU). Early stopping uses a validation split
carved out of the training data only."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from signal_aug.data.loader import train_val_split


class _CNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).squeeze(-1))


class CNN1DClassifier:
    def __init__(
        self,
        epochs: int = 100,
        patience: int = 15,
        batch_size: int = 32,
        lr: float = 1e-3,
        val_fraction: float = 0.2,
        hidden: int = 64,
        seed: int = 0,
    ):
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.val_fraction = val_fraction
        self.hidden = hidden
        self.seed = seed
        self.model: _CNN | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNN1DClassifier":
        torch.manual_seed(self.seed)
        n_classes = int(y.max()) + 1
        X_tr, y_tr, X_val, y_val = train_val_split(X, y, self.val_fraction, self.seed)
        if len(y_val) == 0:  # tiny datasets: fall back to training loss
            X_val, y_val = X_tr, y_tr

        model = _CNN(X.shape[1], n_classes, hidden=self.hidden)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        Xt = torch.from_numpy(X_tr).float()
        yt = torch.from_numpy(y_tr).long()
        Xv = torch.from_numpy(X_val).float()
        yv = torch.from_numpy(y_val).long()

        best_loss, best_state, bad_epochs = float("inf"), None, 0
        gen = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            model.train()
            perm = torch.randperm(len(Xt), generator=gen)
            for start in range(0, len(Xt), self.batch_size):
                idx = perm[start : start + self.batch_size]
                if len(idx) < 2:  # BatchNorm needs >1 sample
                    continue
                opt.zero_grad()
                loss = loss_fn(model(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(Xv), yv))
            if val_loss < best_loss - 1e-5:
                best_loss, bad_epochs = val_loss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() must be called before predict()")
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).float())
        return logits.argmax(dim=1).numpy().astype(np.int64)
