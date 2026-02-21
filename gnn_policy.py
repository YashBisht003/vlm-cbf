from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


@dataclass
class ActionOut:
    action: torch.Tensor  # (..., N, 4) -> tanh(vx,vy,yaw), grip in {0,1}
    pre_tanh: torch.Tensor  # (..., N, 3) pre-tanh values for logprob
    logprob: torch.Tensor  # (..., N)
    grip_action: torch.Tensor  # (..., N)


class GnnPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, msg_dim: int = 128, layers: int = 3) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.msg_dim = msg_dim
        self.layers = layers

        self.encoder = _mlp(obs_dim, hidden, hidden)
        self.msg_mlps = nn.ModuleList(
            [_mlp(hidden * 2 + 1, msg_dim, msg_dim) for _ in range(layers)]
        )
        self.upd_mlps = nn.ModuleList(
            [_mlp(hidden + msg_dim, hidden, hidden) for _ in range(layers)]
        )

        self.mu_head = _mlp(hidden, hidden, 3)
        self.logstd = nn.Parameter(torch.zeros(3))
        self.grip_head = _mlp(hidden, hidden, 1)
        self.min_logstd = -5.0
        self.max_logstd = 1.5

    def forward(self, obs: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (N, obs_dim) or (B, N, obs_dim)
            pos: (N, 2) or (B, N, 2)
        Returns:
            node embeddings (N, hidden) or (B, N, hidden)
        """
        squeeze_batch = False
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
            pos = pos.unsqueeze(0)
            squeeze_batch = True
        elif obs.dim() != 3:
            raise ValueError(f"Unexpected obs shape: {tuple(obs.shape)}")
        elif pos.dim() == 2:
            pos = pos.unsqueeze(0)
        elif pos.dim() != 3:
            raise ValueError(f"Unexpected pos shape: {tuple(pos.shape)}")
        if obs.shape[0] != pos.shape[0] or obs.shape[1] != pos.shape[1]:
            raise ValueError(f"Batch/agent mismatch: obs {tuple(obs.shape)} vs pos {tuple(pos.shape)}")

        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)
        pos = torch.nan_to_num(pos, nan=0.0, posinf=1e3, neginf=-1e3)
        z = self.encoder(obs)
        z = torch.nan_to_num(z, nan=0.0, posinf=1e3, neginf=-1e3)
        batch_size, n_agents = z.shape[0], z.shape[1]
        eye_mask = torch.eye(n_agents, dtype=torch.bool, device=z.device).view(1, n_agents, n_agents, 1)
        for idx in range(self.layers):
            dij = torch.cdist(pos, pos, p=2).unsqueeze(-1)
            dij = torch.nan_to_num(dij, nan=0.0, posinf=1e3, neginf=0.0)
            z_i = z.unsqueeze(2).expand(-1, -1, n_agents, -1)
            z_j = z.unsqueeze(1).expand(-1, n_agents, -1, -1)
            pair_inp = torch.cat([z_i, z_j, dij], dim=-1).reshape(batch_size * n_agents * n_agents, -1)
            pair_msg = self.msg_mlps[idx](pair_inp).reshape(batch_size, n_agents, n_agents, self.msg_dim)
            pair_msg = pair_msg.masked_fill(eye_mask, 0.0)
            msg_tensor = pair_msg.sum(dim=2)

            upd_in = torch.cat([z, msg_tensor], dim=-1).reshape(batch_size * n_agents, -1)
            z = self.upd_mlps[idx](upd_in).reshape(batch_size, n_agents, self.hidden)
            z = torch.nan_to_num(z, nan=0.0, posinf=1e3, neginf=-1e3)
        if squeeze_batch:
            return z.squeeze(0)
        return z

    def _tanh_normal(self, mu: torch.Tensor, logstd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = torch.nan_to_num(mu, nan=0.0, posinf=20.0, neginf=-20.0)
        logstd = torch.clamp(torch.nan_to_num(logstd, nan=0.0, posinf=self.max_logstd, neginf=self.min_logstd), self.min_logstd, self.max_logstd)
        std = logstd.exp()
        eps = torch.randn_like(mu)
        pre_tanh = mu + eps * std
        pre_tanh = torch.nan_to_num(pre_tanh, nan=0.0, posinf=20.0, neginf=-20.0)
        action = torch.tanh(pre_tanh)
        logprob = self._tanh_normal_logprob(mu, logstd, pre_tanh, action)
        return action, pre_tanh, logprob

    @staticmethod
    def _tanh_normal_logprob(
        mu: torch.Tensor, logstd: torch.Tensor, pre_tanh: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        mu = torch.nan_to_num(mu, nan=0.0, posinf=20.0, neginf=-20.0)
        logstd = torch.clamp(torch.nan_to_num(logstd, nan=0.0, posinf=1.5, neginf=-5.0), -5.0, 1.5)
        pre_tanh = torch.nan_to_num(pre_tanh, nan=0.0, posinf=20.0, neginf=-20.0)
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        std = logstd.exp()
        normal = torch.distributions.Normal(mu, std)
        logprob = normal.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        logprob = torch.nan_to_num(logprob, nan=-1e3, posinf=1e3, neginf=-1e3)
        return logprob.sum(dim=-1)

    def act(self, obs: torch.Tensor, pos: torch.Tensor, deterministic: bool = False) -> ActionOut:
        z = self.forward(obs, pos)
        mu = self.mu_head(z)
        mu = torch.clamp(torch.nan_to_num(mu, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
        if deterministic:
            pre_tanh = mu
            action = torch.tanh(pre_tanh)
            logprob = self._tanh_normal_logprob(mu, self.logstd, pre_tanh, action)
        else:
            action, pre_tanh, logprob = self._tanh_normal(mu, self.logstd)
        grip_logits = self.grip_head(z).squeeze(-1)
        grip_logits = torch.nan_to_num(grip_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        if deterministic:
            grip_action = (torch.sigmoid(grip_logits) > 0.5).float()
            grip_logprob = torch.distributions.Bernoulli(logits=grip_logits).log_prob(grip_action)
        else:
            dist = torch.distributions.Bernoulli(logits=grip_logits)
            grip_action = dist.sample()
            grip_logprob = dist.log_prob(grip_action)
        logprob = logprob + grip_logprob
        action_full = torch.cat([action, grip_action.unsqueeze(-1)], dim=-1)
        return ActionOut(action=action_full, pre_tanh=pre_tanh, logprob=logprob, grip_action=grip_action)

    def evaluate_actions(
        self, obs: torch.Tensor, pos: torch.Tensor, pre_tanh: torch.Tensor, grip_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logprob, entropy, _mu = self.evaluate_actions_with_mu(obs, pos, pre_tanh, grip_action)
        return logprob, entropy

    def evaluate_actions_with_mu(
        self, obs: torch.Tensor, pos: torch.Tensor, pre_tanh: torch.Tensor, grip_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.forward(obs, pos)
        mu = self.mu_head(z)
        mu = torch.clamp(torch.nan_to_num(mu, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
        logprob = self._tanh_normal_logprob(mu, self.logstd, pre_tanh, torch.tanh(pre_tanh))
        grip_logits = self.grip_head(z).squeeze(-1)
        grip_logits = torch.nan_to_num(grip_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        grip_logprob = torch.distributions.Bernoulli(logits=grip_logits).log_prob(grip_action)
        logprob = logprob + grip_logprob
        logstd = torch.clamp(torch.nan_to_num(self.logstd, nan=0.0, posinf=self.max_logstd, neginf=self.min_logstd), self.min_logstd, self.max_logstd)
        entropy = torch.distributions.Normal(mu, logstd.exp()).entropy().sum(dim=-1)
        entropy = entropy + torch.distributions.Bernoulli(logits=grip_logits).entropy()
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=1e3, neginf=0.0)
        return logprob, entropy, mu


class CentralCritic(nn.Module):
    def __init__(self, obs_dim: int, n_agents: int = 4, hidden: int = 256, global_dim: int = 0) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.global_dim = int(global_dim)
        self.net = _mlp(obs_dim * n_agents + self.global_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, global_state: torch.Tensor | None = None) -> torch.Tensor:
        # Accept either a single state (N, D) or a batch (B, N, D).
        if obs.dim() == 2:
            if obs.shape[0] != self.n_agents:
                raise ValueError(f"Expected {self.n_agents} agents, got {obs.shape[0]}")
            flat = obs.reshape(1, -1)
        elif obs.dim() == 3:
            if obs.shape[1] != self.n_agents:
                raise ValueError(f"Expected {self.n_agents} agents, got {obs.shape[1]}")
            flat = obs.reshape(obs.shape[0], -1)
        else:
            raise ValueError(f"Unexpected critic input shape: {tuple(obs.shape)}")
        if self.global_dim > 0:
            if global_state is None:
                raise ValueError("global_state is required when global_dim > 0")
            if global_state.dim() == 1:
                g = global_state.reshape(1, -1)
            elif global_state.dim() == 2:
                g = global_state
            else:
                raise ValueError(f"Unexpected global_state shape: {tuple(global_state.shape)}")
            if g.shape[1] != self.global_dim:
                raise ValueError(f"Expected global_state dim {self.global_dim}, got {g.shape[1]}")
            if g.shape[0] not in (1, flat.shape[0]):
                raise ValueError(f"Batch mismatch: critic batch {flat.shape[0]} vs global {g.shape[0]}")
            if g.shape[0] == 1 and flat.shape[0] > 1:
                g = g.expand(flat.shape[0], -1)
            flat = torch.cat([flat, g], dim=1)
        return self.net(flat).squeeze(-1)
