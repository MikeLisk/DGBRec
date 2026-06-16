# -*- coding: utf-8 -*-
"""DGBRec model."""

from typing import Any, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gates import DegreeAwareScalarGate
from .granular_ball import DifferentiableGranularBall
from .spectral import SpectralInitializer


class DGBRec(nn.Module):
    def __init__(self, dataset: Any, config: Any):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_nodes = dataset.n_nodes

        spectral_initializer = SpectralInitializer(
            dim=config.emb_dim,
            n_iter=config.random_svd_n_iter,
            oversampling=config.random_svd_oversampling,
            seed=config.random_svd_seed,
        )
        user_struct, item_struct = spectral_initializer(dataset, config.device)

        self.embedding_user = nn.Embedding(self.n_users, config.emb_dim)
        self.embedding_item = nn.Embedding(self.n_items, config.emb_dim)
        with torch.no_grad():
            self.embedding_user.weight.copy_(user_struct + 0.01 * torch.randn_like(user_struct))
            self.embedding_item.weight.copy_(item_struct + 0.01 * torch.randn_like(item_struct))

        self.fine_graph = dataset.graph
        self.user_ball = DifferentiableGranularBall(
            n_balls=config.n_balls_u,
            dim=config.emb_dim,
            init_feat=user_struct,
            kmeans_iters=config.kmeans_iters,
            radius_init_scale=config.radius_init_scale,
            radius_eps=config.radius_eps,
            radius_min=config.radius_min,
            radius_max=config.radius_max,
        )
        self.item_ball = DifferentiableGranularBall(
            n_balls=config.n_balls_i,
            dim=config.emb_dim,
            init_feat=item_struct,
            kmeans_iters=config.kmeans_iters,
            radius_init_scale=config.radius_init_scale,
            radius_eps=config.radius_eps,
            radius_min=config.radius_min,
            radius_max=config.radius_max,
        )

        self.user_gate = DegreeAwareScalarGate(config.emb_dim)
        self.item_gate = DegreeAwareScalarGate(config.emb_dim)
        self.layer_logits = nn.Parameter(torch.zeros(config.n_layers + 1))
        self.register_buffer("norm_log_degree", dataset.norm_log_degree.clone())

    def _aggregate_to_balls(
        self,
        H_u: torch.Tensor,
        H_i: torch.Tensor,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mass_u = H_u.sum(dim=0).view(-1, 1)
        mass_i = H_i.sum(dim=0).view(-1, 1)
        ball_u_proto = torch.mm(H_u.t(), user_emb) / (mass_u + 1e-8)
        ball_i_proto = torch.mm(H_i.t(), item_emb) / (mass_i + 1e-8)
        return ball_u_proto, ball_i_proto, mass_u.squeeze(1), mass_i.squeeze(1)

    def _row_purity(self, mat: torch.Tensor, num_cols: int) -> torch.Tensor:
        row_sum = mat.sum(dim=1, keepdim=True)
        valid = (row_sum.squeeze(1) > 1e-8).float()
        if num_cols <= 1:
            return valid
        prob = mat / (row_sum + 1e-8)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
        norm_entropy = entropy / math.log(num_cols)
        purity = 1.0 - norm_entropy
        return torch.clamp(purity * valid, min=0.0, max=1.0)

    def _compute_collaborative_purity_and_structure(
        self,
        H_u: torch.Tensor,
        H_i: torch.Tensor,
        mass_u: torch.Tensor,
        mass_i: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        R = self.dataset.interaction_sparse
        RH_i = torch.sparse.mm(R, H_i)
        P = torch.mm(H_u.t(), RH_i)

        q_u = self._row_purity(P, self.item_ball.n_balls)
        q_i = self._row_purity(P.t(), self.user_ball.n_balls)

        omega_u = torch.log1p(mass_u) / (torch.max(torch.log1p(mass_u)) + 1e-8)
        omega_i = torch.log1p(mass_i) / (torch.max(torch.log1p(mass_i)) + 1e-8)
        gamma_u = q_u * omega_u
        gamma_i = q_i * omega_i

        row_deg = P.sum(dim=1)
        col_deg = P.sum(dim=0)
        P_norm = P / (torch.sqrt(row_deg.view(-1, 1) + 1e-8) * torch.sqrt(col_deg.view(1, -1) + 1e-8) + 1e-8)
        return q_u, q_i, gamma_u, gamma_i, P_norm

    def _ball_attention_and_update(
        self,
        ball_u_proto: torch.Tensor,
        ball_i_proto: torch.Tensor,
        gamma_u: torch.Tensor,
        gamma_i: torch.Tensor,
        P_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = ball_u_proto.size(1)
        gamma_u_eff = self.config.q_floor + (1.0 - self.config.q_floor) * torch.clamp(gamma_u, 0.0, 1.0)
        gamma_i_eff = self.config.q_floor + (1.0 - self.config.q_floor) * torch.clamp(gamma_i, 0.0, 1.0)

        score_u_i = torch.mm(ball_u_proto, ball_i_proto.t()) / math.sqrt(dim)
        score_u_i = score_u_i + self.config.eta_struct * torch.log(P_norm + 1e-8)
        score_u_i = score_u_i + self.config.beta_quality * torch.log(gamma_i_eff.view(1, -1) + 1e-8)
        A_u_from_i = F.softmax(score_u_i, dim=1)
        Z_u_tmp = torch.mm(A_u_from_i, ball_i_proto)

        score_i_u = torch.mm(ball_i_proto, ball_u_proto.t()) / math.sqrt(dim)
        score_i_u = score_i_u + self.config.eta_struct * torch.log(P_norm.t() + 1e-8)
        score_i_u = score_i_u + self.config.beta_quality * torch.log(gamma_u_eff.view(1, -1) + 1e-8)
        A_i_from_u = F.softmax(score_i_u, dim=1)
        Z_i_tmp = torch.mm(A_i_from_u, ball_u_proto)

        Z_u = gamma_u_eff.view(-1, 1) * Z_u_tmp + (1.0 - gamma_u_eff.view(-1, 1)) * ball_u_proto
        Z_i = gamma_i_eff.view(-1, 1) * Z_i_tmp + (1.0 - gamma_i_eff.view(-1, 1)) * ball_i_proto
        return Z_u, Z_i, A_u_from_i, A_i_from_u

    def forward(self, current_tau_h: float, return_aux: bool = False):
        all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight], dim=0)
        layer_embs = [all_emb]

        prev_user_centers = self.user_ball.centers
        prev_item_centers = self.item_ball.centers
        prev_user_radii = self.user_ball.get_base_radius()
        prev_item_radii = self.item_ball.get_base_radius()
        last_aux = None

        for layer_idx in range(self.config.n_layers):
            curr_user_emb = all_emb[:self.n_users]
            curr_item_emb = all_emb[self.n_users:]

            # Fine-grained LightGCN-style propagation
            fine_emb = torch.sparse.mm(self.fine_graph, all_emb)
            fine_user = fine_emb[:self.n_users]
            fine_item = fine_emb[self.n_users:]

            # Coarse-grained granular-ball propagation
            H_u, user_centers, user_radii = self.user_ball.layer_adaptive_assignment(
                feats=curr_user_emb,
                prev_centers=prev_user_centers,
                prev_radii=prev_user_radii,
                tau_h=current_tau_h,
                top_m=self.config.topm_membership,
                rho_c=self.config.rho_c,
                rho_r=self.config.rho_r,
                detach_update=self.config.detach_ball_update,
            )
            H_i, item_centers, item_radii = self.item_ball.layer_adaptive_assignment(
                feats=curr_item_emb,
                prev_centers=prev_item_centers,
                prev_radii=prev_item_radii,
                tau_h=current_tau_h,
                top_m=self.config.topm_membership,
                rho_c=self.config.rho_c,
                rho_r=self.config.rho_r,
                detach_update=self.config.detach_ball_update,
            )

            ball_u_proto, ball_i_proto, mass_u, mass_i = self._aggregate_to_balls(H_u, H_i, curr_user_emb, curr_item_emb)
            q_u, q_i, gamma_u, gamma_i, P_norm = self._compute_collaborative_purity_and_structure(
                H_u, H_i, mass_u, mass_i
            )
            Z_ball_u, Z_ball_i, A_u_from_i, A_i_from_u = self._ball_attention_and_update(
                ball_u_proto, ball_i_proto, gamma_u, gamma_i, P_norm
            )
            coarse_user = torch.mm(H_u, Z_ball_u)
            coarse_item = torch.mm(H_i, Z_ball_i)

            # Degree-aware dual-granularity fusion
            norm_deg_user = self.norm_log_degree[:self.n_users]
            norm_deg_item = self.norm_log_degree[self.n_users:]
            gate_user = self.user_gate(fine_user, coarse_user, norm_deg_user)
            gate_item = self.item_gate(fine_item, coarse_item, norm_deg_item)
            next_user_emb = gate_user * fine_user + (1.0 - gate_user) * coarse_user
            next_item_emb = gate_item * fine_item + (1.0 - gate_item) * coarse_item

            all_emb = torch.cat([next_user_emb, next_item_emb], dim=0)
            layer_embs.append(all_emb)

            last_aux = {
                "layer_idx": layer_idx,
                "H_u": H_u,
                "H_i": H_i,
                "q_u": q_u,
                "q_i": q_i,
                "gamma_u": gamma_u,
                "gamma_i": gamma_i,
                "user_centers": user_centers,
                "item_centers": item_centers,
                "user_radii": user_radii,
                "item_radii": item_radii,
                "ball_u_proto": ball_u_proto,
                "ball_i_proto": ball_i_proto,
                "layer_user_emb": curr_user_emb,
                "layer_item_emb": curr_item_emb,
                "gate_user": gate_user,
                "gate_item": gate_item,
                "A_u_from_i": A_u_from_i,
                "A_i_from_u": A_i_from_u,
            }

            prev_user_centers = user_centers
            prev_item_centers = item_centers
            prev_user_radii = user_radii
            prev_item_radii = item_radii

        layer_stack = torch.stack(layer_embs, dim=1)
        layer_weights = F.softmax(self.layer_logits, dim=0)
        final_emb = torch.sum(layer_stack * layer_weights.view(1, -1, 1), dim=1)
        user_final, item_final = torch.split(final_emb, [self.n_users, self.n_items], dim=0)

        if return_aux:
            return user_final, item_final, last_aux
        return user_final, item_final

    def _granular_ball_quality_loss(self, batch_users: torch.Tensor, batch_pos_items: torch.Tensor, aux: Dict[str, Any]) -> torch.Tensor:
        H_u = aux["H_u"]
        H_i = aux["H_i"]
        q_u = aux["q_u"]
        q_i = aux["q_i"]
        user_centers = aux["user_centers"]
        item_centers = aux["item_centers"]
        user_radii = aux["user_radii"]
        item_radii = aux["item_radii"]
        layer_user_emb = aux["layer_user_emb"]
        layer_item_emb = aux["layer_item_emb"]

        batch_user_feat = layer_user_emb[batch_users]
        batch_item_feat = layer_item_emb[batch_pos_items]
        batch_H_u = H_u[batch_users]
        batch_H_i = H_i[batch_pos_items]

        loss_cov_u = self.user_ball.coverage_loss(
            feats=batch_user_feat,
            assignment=batch_H_u,
            centers=user_centers,
            radii=user_radii,
            coverage_margin=self.config.coverage_margin,
        )
        loss_cov_i = self.item_ball.coverage_loss(
            feats=batch_item_feat,
            assignment=batch_H_i,
            centers=item_centers,
            radii=item_radii,
            coverage_margin=self.config.coverage_margin,
        )
        loss_cov = loss_cov_u + loss_cov_i
        loss_rad = self.user_ball.radius_regularization(user_radii) + self.item_ball.radius_regularization(item_radii)
        loss_purity = 0.5 * (torch.mean(1.0 - q_u) + torch.mean(1.0 - q_i))
        loss_mass = self.user_ball.mass_balance_loss(H_u) + self.item_ball.mass_balance_loss(H_i)

        loss_gb = (
            self.config.lambda_cov * loss_cov
            + self.config.lambda_rad * loss_rad
            + self.config.lambda_purity * loss_purity
            + self.config.lambda_mass * loss_mass
        )
        return loss_gb

    def _regularization_loss(self, batch_users: torch.Tensor, batch_pos_items: torch.Tensor, batch_neg_items: torch.Tensor) -> torch.Tensor:
        batch_size = max(1, batch_users.numel())
        u_emb_w = self.embedding_user.weight[batch_users]
        pos_emb_w = self.embedding_item.weight[batch_pos_items]
        if batch_neg_items.dim() == 1:
            neg_emb_w = self.embedding_item.weight[batch_neg_items]
            neg_norm = neg_emb_w.norm(2).pow(2)
        else:
            neg_emb_w = self.embedding_item.weight[batch_neg_items]
            neg_norm = neg_emb_w.norm(2).pow(2) / max(1, batch_neg_items.size(1))
        emb_reg = (u_emb_w.norm(2).pow(2) + pos_emb_w.norm(2).pow(2) + neg_norm) / batch_size

        ball_reg = (
            self.user_ball.centers.norm(2).pow(2) / max(1, self.user_ball.n_balls)
            + self.item_ball.centers.norm(2).pow(2) / max(1, self.item_ball.n_balls)
            + self.user_ball.get_base_radius().norm(2).pow(2) / max(1, self.user_ball.n_balls)
            + self.item_ball.get_base_radius().norm(2).pow(2) / max(1, self.item_ball.n_balls)
        )

        gate_reg = torch.tensor(0.0, device=self.config.device)
        for p in self.user_gate.parameters():
            gate_reg += p.norm(2).pow(2)
        for p in self.item_gate.parameters():
            gate_reg += p.norm(2).pow(2)
        layer_reg = self.layer_logits.norm(2).pow(2)

        reg_loss = self.config.decay * (emb_reg + ball_reg + 1e-4 * gate_reg + 1e-4 * layer_reg)
        return reg_loss

    @staticmethod
    def _ranking_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        return -torch.mean(F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores))

    def calculate_loss(
        self,
        batch_users: torch.Tensor,
        batch_pos_items: torch.Tensor,
        batch_neg_items: torch.Tensor,
        current_tau_h: float,
        gb_weight: float = 1.0,
    ) -> torch.Tensor:
        user_final, item_final, aux = self.forward(current_tau_h=current_tau_h, return_aux=True)
        batch_user_emb = user_final[batch_users]
        batch_pos_emb = item_final[batch_pos_items]
        if batch_neg_items.dim() == 1:
            batch_neg_items = batch_neg_items.view(-1, 1)
        batch_neg_emb = item_final[batch_neg_items]
        pos_scores = torch.sum(batch_user_emb * batch_pos_emb, dim=1)
        neg_scores = torch.sum(batch_user_emb.unsqueeze(1) * batch_neg_emb, dim=-1)
        loss_bpr = self._ranking_loss(pos_scores, neg_scores)
        # loss_gb = self._granular_ball_quality_loss(batch_users, batch_pos_items, aux)
        reg_loss = self._regularization_loss(batch_users, batch_pos_items, batch_neg_items)
        return loss_bpr  + reg_loss
