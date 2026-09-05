"""Class-conditional 1D-convolutional VAE augmenter (issue #42, task GN-A1).

Unlike every other augmenter here, this one LEARNS a generator before it can
sample. The fit happens inside the augment call, on exactly the array it was
handed -- which is always a training split (see the module docstring of
methods.py and spec section 8). Nothing is cached or shared between calls, so a
generator can never see data belonging to another run's split. That is the
whole reason the fit is not hoisted out for speed.

Architecture is a small TimeVAE-style conv VAE: two strided convolutions down,
a Gaussian latent, two convolutions back up, then a linear resize to the exact
input length so any series length works (24 for ItalyPowerDemand, 500 for
FordA). The class label is fed to both encoder and decoder as one-hot channels,
making it a plain CVAE: sampling the prior with a fixed label draws from that
class's learned distribution.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _halved(length: int) -> int:
    """Output length of Conv1d(kernel=3, stride=2, padding=1)."""
    return (length + 1) // 2


class _CVAE(nn.Module):
    def __init__(self, n_channels: int, length: int, n_classes: int, latent_dim: int, hidden: int = 32):
        super().__init__()
        self.n_channels, self.length, self.n_classes = n_channels, length, n_classes
        self.hidden = hidden
        # decoder starts from a quarter-length feature map, mirroring the two
        # strided encoder convolutions; floored at 4 so very short series still
        # have something to convolve over
        self.reduced = max(4, _halved(_halved(length)))

        self.enc = nn.Sequential(
            nn.Conv1d(n_channels + n_classes, hidden, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden * 2, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        enc_flat = hidden * 2 * _halved(_halved(length))
        self.to_mu = nn.Linear(enc_flat, latent_dim)
        self.to_logvar = nn.Linear(enc_flat, latent_dim)

        self.from_z = nn.Linear(latent_dim + n_classes, hidden * 2 * self.reduced)
        self.dec = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, n_channels, 3, padding=1),
        )

    def _label_channels(self, y1h: torch.Tensor, length: int) -> torch.Tensor:
        return y1h[:, :, None].expand(-1, -1, length)

    def encode(self, x: torch.Tensor, y1h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([x, self._label_channels(y1h, x.shape[-1])], dim=1)
        h = self.enc(h).flatten(1)
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z: torch.Tensor, y1h: torch.Tensor) -> torch.Tensor:
        h = self.from_z(torch.cat([z, y1h], dim=1))
        h = h.view(-1, self.hidden * 2, self.reduced)
        h = self.dec(h)
        return F.interpolate(h, size=self.length, mode="linear", align_corners=False)

    def forward(self, x: torch.Tensor, y1h: torch.Tensor):
        mu, logvar = self.encode(x, y1h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z, y1h), mu, logvar


def fit_and_sample(
    X: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
    y_new_idx: np.ndarray,
    seed: int,
    latent_dim: int,
    steps: int,
    beta: float,
    lr: float,
    batch_size: int,
) -> np.ndarray:
    """Train a CVAE on (X, y_idx) and sample one series per entry of y_new_idx.

    `steps` is the number of optimizer updates (not epochs).

    y_idx / y_new_idx are contiguous class indices (0..n_classes-1), not the
    original labels; the caller maps them back.
    """
    torch.manual_seed(seed)
    n, n_channels, length = X.shape

    model = _CVAE(n_channels, length, n_classes, latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xt = torch.from_numpy(np.ascontiguousarray(X)).float()
    y1h = F.one_hot(torch.from_numpy(y_idx).long(), num_classes=n_classes).float()

    gen = torch.Generator().manual_seed(seed)
    model.train()
    # A fixed budget of optimizer STEPS, not epochs: steps-per-epoch scales
    # with the training-set size, so an epoch budget would silently give a
    # 20-sample dataset a thirtieth of the training a 3600-sample one gets.
    # Steps make the generator's training budget comparable across datasets
    # and its cost predictable.
    done = 0
    while done < steps:
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            if done >= steps:
                break
            idx = perm[start : start + batch_size]
            xb, yb = Xt[idx], y1h[idx]
            opt.zero_grad()
            recon, mu, logvar = model(xb, yb)
            # summed over channels/time, averaged over the batch: keeps the
            # reconstruction/KL balance independent of series length
            recon_loss = F.mse_loss(recon, xb, reduction="none").flatten(1).sum(1).mean()
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
            (recon_loss + beta * kl).backward()
            opt.step()
            done += 1

    model.eval()
    out = []
    y_new_1h = F.one_hot(torch.from_numpy(y_new_idx).long(), num_classes=n_classes).float()
    with torch.no_grad():
        for start in range(0, len(y_new_idx), batch_size):
            yb = y_new_1h[start : start + batch_size]
            z = torch.randn(len(yb), model.to_mu.out_features, generator=gen)
            out.append(model.decode(z, yb).numpy())
    return np.concatenate(out, axis=0).astype(np.float32)
