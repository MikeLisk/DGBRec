# -*- coding: utf-8 -*-
"""Degree-aware scalar gate for dual-granularity fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DegreeAwareScalarGate(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.linear = nn.Linear(emb_dim * 2, 1)
        self.raw_degree_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, fine_emb: torch.Tensor, coarse_emb: torch.Tensor, norm_log_degree: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([fine_emb, coarse_emb], dim=1)
        logits = self.linear(combined)
        degree_alpha = F.softplus(self.raw_degree_alpha)
        degree_bias = degree_alpha * norm_log_degree.view(-1, 1)
        gate = torch.sigmoid(logits + degree_bias)
        return gate
