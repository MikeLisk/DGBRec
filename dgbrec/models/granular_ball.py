# -*- coding: utf-8 -*-
"""Differentiable radius-aware granular ball module."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableGranularBall(nn.Module):
    def __init__(
        self,
        n_balls: int,
        dim: int,
        init_feat: torch.Tensor,
        kmeans_iters: int = 8,
        radius_init_scale: float = 1.0,
        radius_eps: float = 1e-6,
        radius_min: float = 1e-4,
        radius_max: float = 10.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_nodes = init_feat.size(0)
        self.n_balls = min(max(1, int(n_balls)), self.n_nodes)
        self.radius_eps = radius_eps
        self.radius_min = radius_min
        self.radius_max = radius_max

        with torch.no_grad():
            centers, assignments = self._run_kmeans(init_feat, self.n_balls, kmeans_iters)
            init_radius = self._estimate_initial_radii(init_feat, centers, assignments, radius_init_scale)
            self.centers = nn.Parameter(centers.clone())
            rho_init = self._inverse_softplus(init_radius - self.radius_eps)
            self.rho_radius = nn.Parameter(rho_init.clone())

    @staticmethod
    def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        y_norm = (y ** 2).sum(dim=1, keepdim=True).t()
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.clamp(dist, min=0.0)

    @staticmethod
    def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
        y = torch.clamp(y, min=1e-8)
        return torch.log(torch.expm1(y))

    def _run_kmeans(self, feats: torch.Tensor, n_clusters: int, iters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = feats.detach()
        n = feats.size(0)
        device = feats.device
        perm = torch.randperm(n, device=device)
        centers = feats[perm[:n_clusters]].clone()
        chunk_size = 8192
        assignments = torch.zeros(n, dtype=torch.long, device=device)

        for _ in range(iters):
            assign_chunks = []
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                dist = self._pairwise_sq_dist(feats[start:end], centers)
                assign_chunks.append(torch.argmin(dist, dim=1))
            assignments = torch.cat(assign_chunks, dim=0)

            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(n_clusters, 1, device=device)
            new_centers.index_add_(0, assignments, feats)
            counts.index_add_(0, assignments, torch.ones(n, 1, device=device))

            non_empty = counts.squeeze(1) > 0
            new_centers[non_empty] = new_centers[non_empty] / counts[non_empty]
            empty = ~non_empty
            if empty.any():
                empty_num = int(empty.sum().item())
                refill_idx = torch.randperm(n, device=device)[:empty_num]
                new_centers[empty] = feats[refill_idx]
            centers = new_centers
        return centers, assignments

    def _estimate_initial_radii(
        self,
        feats: torch.Tensor,
        centers: torch.Tensor,
        assignments: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        device = feats.device
        n_clusters = centers.size(0)
        dist2_sum = torch.zeros(n_clusters, device=device)
        counts = torch.zeros(n_clusters, device=device)
        diff = feats - centers[assignments]
        dist2 = torch.clamp((diff ** 2).sum(dim=1), min=0.0)
        dist2_sum.index_add_(0, assignments, dist2)
        counts.index_add_(0, assignments, torch.ones_like(dist2))

        valid = counts > 0
        global_radius = torch.sqrt(dist2.mean() + self.radius_eps)
        radii = torch.ones(n_clusters, device=device) * global_radius
        radii[valid] = torch.sqrt(dist2_sum[valid] / (counts[valid] + self.radius_eps) + self.radius_eps)
        radii = torch.clamp(radii * scale, min=self.radius_min, max=self.radius_max)
        return radii

    def get_base_radius(self) -> torch.Tensor:
        radius = F.softplus(self.rho_radius) + self.radius_eps
        return torch.clamp(radius, min=self.radius_min, max=self.radius_max)

    def normalized_distance(
        self,
        feats: torch.Tensor,
        centers: Optional[torch.Tensor] = None,
        radii: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if centers is None:
            centers = self.centers
        if radii is None:
            radii = self.get_base_radius()
        dist2 = self._pairwise_sq_dist(feats, centers)
        return dist2 / (radii.pow(2).unsqueeze(0) + self.radius_eps)

    def soft_membership(
        self,
        feats: torch.Tensor,
        centers: torch.Tensor,
        radii: torch.Tensor,
        tau_h: float = 1.0,
        top_m: Optional[int] = None,
    ) -> torch.Tensor:
        delta = self.normalized_distance(feats, centers, radii)
        logits = -delta / max(float(tau_h), self.radius_eps)
        probs = F.softmax(logits, dim=1)
        if top_m is not None and top_m > 0 and top_m < self.n_balls:
            _, top_idx = torch.topk(probs, k=top_m, dim=1)
            mask = torch.zeros_like(probs).scatter_(1, top_idx, 1.0)
            sparse_probs = probs * mask
            sparse_probs = sparse_probs / (sparse_probs.sum(dim=1, keepdim=True) + self.radius_eps)
            return sparse_probs
        return probs

    def layer_adaptive_assignment(
        self,
        feats: torch.Tensor,
        prev_centers: torch.Tensor,
        prev_radii: torch.Tensor,
        tau_h: float,
        top_m: int,
        rho_c: float,
        rho_r: float,
        detach_update: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H_bar = self.soft_membership(
            feats=feats,
            centers=prev_centers,
            radii=prev_radii,
            tau_h=tau_h,
            top_m=None,
        )
        mass_bar = H_bar.sum(dim=0)
        mass_bar_safe = mass_bar + self.radius_eps
        refreshed_centers = torch.mm(H_bar.t(), feats) / mass_bar_safe.view(-1, 1)
        dist2_to_refreshed = self._pairwise_sq_dist(feats, refreshed_centers)
        refreshed_radius = torch.sqrt(
            torch.sum(H_bar * dist2_to_refreshed, dim=0) / mass_bar_safe + self.radius_eps
        )
        refreshed_radius = torch.clamp(refreshed_radius, min=self.radius_min, max=self.radius_max)

        if detach_update:
            refreshed_centers = refreshed_centers.detach()
            refreshed_radius = refreshed_radius.detach()

        new_centers = rho_c * prev_centers + (1.0 - rho_c) * refreshed_centers
        new_radii = rho_r * prev_radii + (1.0 - rho_r) * refreshed_radius
        new_radii = torch.clamp(new_radii, min=self.radius_min, max=self.radius_max)

        H = self.soft_membership(
            feats=feats,
            centers=new_centers,
            radii=new_radii,
            tau_h=tau_h,
            top_m=top_m,
        )
        return H, new_centers, new_radii

    def coverage_loss(
        self,
        feats: torch.Tensor,
        assignment: torch.Tensor,
        centers: torch.Tensor,
        radii: torch.Tensor,
        coverage_margin: float = 1.0,
    ) -> torch.Tensor:
        delta = self.normalized_distance(feats, centers, radii)
        violation = F.relu(delta - coverage_margin)
        loss = torch.sum(assignment * violation) / (torch.sum(assignment) + self.radius_eps)
        return loss

    def radius_regularization(self, radii: torch.Tensor) -> torch.Tensor:
        return torch.mean(radii.pow(2))

    def mass_balance_loss(self, assignment: torch.Tensor) -> torch.Tensor:
        mass = assignment.sum(dim=0)
        prob = mass / (mass.sum() + self.radius_eps)
        target = torch.ones_like(prob) / prob.numel()
        loss = torch.mean((prob - target).pow(2)) * prob.numel()
        return loss
