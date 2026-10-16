"""Checkpoint-compatible neural architectures for the final model.

Source: ``model`` commit 9f3e0a4, ``nets.py``. Layer order and names are kept
exactly, so historical state dictionaries load with ``strict=True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .state import STATE_DIM


def kl_loss(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Mean KL divergence from the unit Gaussian, as used to train APAE."""
    variance = sigma.pow(2)
    return 0.5 * torch.mean(
        torch.sum(mu.pow(2) + variance - torch.log(variance + 1.0e-8) - 1.0, dim=1)
    )


class APAE(nn.Module):
    """Dense two-branch autoencoder used by the z24/Ncol checkpoint."""

    def __init__(self, y_a_dim: int, y_t_dim: int, latent_dim: int, n_aux: int = 0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_aux = n_aux
        h_a1, h_a2 = 320, 160
        h_t1, h_t2 = 384, 192
        h_j1, h_j2 = 192, 96
        self.enc_a = nn.Sequential(
            nn.Linear(y_a_dim, h_a1), nn.ELU(), nn.Linear(h_a1, h_a2), nn.ELU()
        )
        self.enc_t = nn.Sequential(
            nn.Linear(y_t_dim, h_t1), nn.ELU(), nn.Linear(h_t1, h_t2), nn.ELU()
        )
        self.enc_joint = nn.Sequential(
            nn.Linear(h_a2 + h_t2 + n_aux, h_j1), nn.ELU(), nn.Linear(h_j1, h_j2), nn.ELU()
        )
        self.to_mu = nn.Linear(h_j2, latent_dim)
        self.to_sigma_pre = nn.Linear(h_j2, latent_dim)
        self.dec_joint = nn.Sequential(
            nn.Linear(latent_dim + n_aux, h_j2),
            nn.ELU(),
            nn.Linear(h_j2, h_j1),
            nn.ELU(),
            nn.Linear(h_j1, h_a2 + h_t2),
            nn.ELU(),
        )
        self.dec_a = nn.Sequential(nn.Linear(h_a2, h_a1), nn.ELU(), nn.Linear(h_a1, y_a_dim))
        self.dec_t = nn.Sequential(nn.Linear(h_t2, h_t1), nn.ELU(), nn.Linear(h_t1, y_t_dim))
        self._split_a = h_a2

    def encode(
        self, y_a: torch.Tensor, y_t: torch.Tensor, aux: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [self.enc_a(y_a), self.enc_t(y_t)]
        if self.n_aux > 0:
            if aux is None:
                raise ValueError("APAE checkpoint requires auxiliary input")
            parts.append(aux)
        hidden = self.enc_joint(torch.cat(parts, dim=-1))
        mu = self.to_mu(hidden)
        sigma = nn.functional.elu(self.to_sigma_pre(hidden)) + 1.0 + 1.0e-5
        return mu, sigma

    def reparam(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return mu + sigma * torch.randn_like(sigma) if self.training else mu

    def decode(
        self, z: torch.Tensor, aux: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_aux > 0 and aux is None:
            raise ValueError("APAE checkpoint requires auxiliary input")
        z_input = torch.cat([z, aux], dim=-1) if self.n_aux > 0 else z
        hidden = self.dec_joint(z_input)
        return self.dec_a(hidden[:, : self._split_a]), self.dec_t(hidden[:, self._split_a :])

    def forward(
        self, y_a: torch.Tensor, y_t: torch.Tensor, aux: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self.encode(y_a, y_t, aux=aux)
        y_a_hat, y_t_hat = self.decode(self.reparam(mu, sigma), aux=aux)
        return y_a_hat, y_t_hat, mu, sigma


class ResidualStepNetwork(nn.Module):
    """The historical R2A network reused as a shared learned update rule."""

    def __init__(
        self, in_dim: int, latent_dim: int, hidden: tuple[int, ...] = (512, 256, 128)
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = in_dim
        for next_width in hidden:
            layers.extend((nn.Linear(width, next_width), nn.ELU()))
            width = next_width
        self.backbone = nn.Sequential(*layers)
        self.to_mu = nn.Linear(width, latent_dim)
        self.to_sigma_pre = nn.Linear(width, latent_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(inputs)
        mu = self.to_mu(hidden)
        sigma = nn.functional.elu(self.to_sigma_pre(hidden)) + 1.0 + 1.0e-5
        return mu, sigma


def load_apae(path: str | Path, device: torch.device | str = "cpu") -> tuple[APAE, dict[str, Any]]:
    """Load and freeze a dense, full-state historical APAE checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("decoder", "denso") != "denso":
        raise ValueError("The final z24/Ncol model requires the dense APAE decoder")
    if checkpoint.get("state") != "full" or checkpoint["y_a_dim"] != STATE_DIM:
        raise ValueError("The final checkpoint must decode the 818 component Ncol state")
    if checkpoint["latent_dim"] != 24 or checkpoint["y_t_dim"] != 3 * 2145:
        raise ValueError("The final APAE must use z24 and 2145 TUD channels")
    if checkpoint["n_aux"] != 0:
        raise ValueError("The final APAE decoder has no auxiliary input")
    model = APAE(
        checkpoint["y_a_dim"], checkpoint["y_t_dim"], checkpoint["latent_dim"], checkpoint["n_aux"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, checkpoint
