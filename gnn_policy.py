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
    action: torch.Tensor  # (N, 4) -> tanh(vx,vy,yaw), grip in {0,1}
    pre_tanh: torch.Tensor  # (N, 3) pre-tanh values for logprob
    logprob: torch.Tensor  # (N,)
    grip_action: torch.Tensor  # (N,)


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

    def forward(self, obs: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (N, obs_dim)
            pos: (N, 2)
        Returns:
            node embeddings (N, hidden)
        """
        z = self.encoder(obs)
        n_agents = z.shape[0]
        for idx in range(self.layers):
            msgs = []
            for i in range(n_agents):
                m_sum = torch.zeros(self.msg_dim, dtype=z.dtype, device=z.device)
                for j in range(n_agents):
                    if i == j:
                        continue
                    dij = torch.norm(pos[i] - pos[j]).unsqueeze(0)
                    inp = torch.cat([z[i], z[j], dij], dim=0)
                    m_ij = self.msg_mlps[idx](inp)
                    m_sum = m_sum + m_ij
                msgs.append(m_sum)
            msg_tensor = torch.stack(msgs, dim=0)
            z = self.upd_mlps[idx](torch.cat([z, msg_tensor], dim=1))
        return z

    def _tanh_normal(self, mu: torch.Tensor, logstd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        std = logstd.exp()
        eps = torch.randn_like(mu)
        pre_tanh = mu + eps * std
        action = torch.tanh(pre_tanh)
        logprob = self._tanh_normal_logprob(mu, logstd, pre_tanh, action)
        return action, pre_tanh, logprob

    @staticmethod
    def _tanh_normal_logprob(
        mu: torch.Tensor, logstd: torch.Tensor, pre_tanh: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        std = logstd.exp()
        normal = torch.distributions.Normal(mu, std)
        logprob = normal.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        return logprob.sum(dim=-1)

    def act(self, obs: torch.Tensor, pos: torch.Tensor, deterministic: bool = False) -> ActionOut:
        z = self.forward(obs, pos)
        mu = self.mu_head(z)
        if deterministic:
            pre_tanh = mu
            action = torch.tanh(pre_tanh)
            logprob = self._tanh_normal_logprob(mu, self.logstd, pre_tanh, action)
        else:
            action, pre_tanh, logprob = self._tanh_normal(mu, self.logstd)
        grip_logits = self.grip_head(z).squeeze(-1)
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
        z = self.forward(obs, pos)
        mu = self.mu_head(z)
        logprob = self._tanh_normal_logprob(mu, self.logstd, pre_tanh, torch.tanh(pre_tanh))
        grip_logits = self.grip_head(z).squeeze(-1)
        grip_logprob = torch.distributions.Bernoulli(logits=grip_logits).log_prob(grip_action)
        logprob = logprob + grip_logprob
        entropy = torch.distributions.Normal(mu, self.logstd.exp()).entropy().sum(dim=-1)
        entropy = entropy + torch.distributions.Bernoulli(logits=grip_logits).entropy()
        return logprob, entropy


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
