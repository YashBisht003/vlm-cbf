from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cov_upper(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float32)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Expected square covariance, got {cov.shape}")
    idx = np.triu_indices(cov.shape[0])
    return cov[idx]


def build_neural_cbf_input(force_n: float, belief_mu: np.ndarray, belief_cov: np.ndarray) -> np.ndarray:
    mu = np.asarray(belief_mu, dtype=np.float32).reshape(-1)
    cov_upper = _cov_upper(np.asarray(belief_cov, dtype=np.float32))
    return np.concatenate([np.array([float(force_n)], dtype=np.float32), mu, cov_upper], axis=0)


class NeuralForceCBF(nn.Module):
    """
    Learned barrier: h_phi([F_i, mu_b, Sigma_b]).
    Output > 0 means safer; output < 0 means riskier.
    """

    def __init__(self, input_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.bias, 0.2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass
class NeuralCbfEval:
    h_value: float
    features: np.ndarray


class NeuralCbfRuntime:
    def __init__(
        self,
        mu_dim: int,
        hidden: int = 64,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.mu_dim = int(mu_dim)
        self.cov_upper_dim = int(self.mu_dim * (self.mu_dim + 1) / 2)
        self.input_dim = 1 + self.mu_dim + self.cov_upper_dim
        self.device = torch.device(device)
        self.model = NeuralForceCBF(input_dim=self.input_dim, hidden=hidden).to(self.device)
        self.model.eval()
        self.has_trained_weights = False
        if model_path:
            self.load(model_path)

    def eval_barrier(self, force_n: float, belief_mu: np.ndarray, belief_cov: np.ndarray) -> NeuralCbfEval:
        features = build_neural_cbf_input(force_n=force_n, belief_mu=belief_mu, belief_cov=belief_cov)
        if features.shape[0] != self.input_dim:
            raise ValueError(f"Neural CBF input dim mismatch: got {features.shape[0]}, expected {self.input_dim}")
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            h_value = float(self.model(x).item())
        return NeuralCbfEval(h_value=h_value, features=features)

    def load(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Neural CBF checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.device)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        self.model.load_state_dict(state)
        self.model.eval()
        self.has_trained_weights = True

    def save(self, model_path: str) -> None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "input_dim": self.input_dim}, path)


def neural_cbf_loss(h_values: torch.Tensor, safe_labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """
    Supervised barrier loss:
    - safe samples (label=1) should satisfy h >= +margin
    - unsafe samples (label=0) should satisfy h <= -margin
    """
    safe_labels = safe_labels.float()
    targets = (safe_labels * 2.0 - 1.0) * float(margin)
    return F.relu(targets - h_values).pow(2).mean()


def neural_cbf_temporal_loss(
    h_prev: torch.Tensor,
    h_next: torch.Tensor,
    alpha_dt: float,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Barrier residual objective:
      h_{t+1} - h_t + alpha*dt*h_t >= margin
    """
    residual = h_next - h_prev + float(alpha_dt) * h_prev
    return F.relu(float(margin) - residual).pow(2).mean()
