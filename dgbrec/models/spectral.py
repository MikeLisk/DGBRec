# -*- coding: utf-8 -*-
"""Randomized SVD spectral initialization."""

from typing import Any, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


class SpectralInitializer:
    def __init__(self, dim: int, n_iter: int = 3, oversampling: int = 16, seed: int = 2025):
        self.dim = dim
        self.n_iter = n_iter
        self.oversampling = oversampling
        self.seed = seed

    def _randomized_svd_sparse(self, A: sp.csr_matrix, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        n_rows, n_cols = A.shape
        max_rank = max(1, min(n_rows, n_cols) - 1)
        k = min(k, max_rank)
        sample_dim = min(k + self.oversampling, max_rank)

        omega = rng.standard_normal(size=(n_cols, sample_dim)).astype(np.float32)
        Y = A.dot(omega)
        for _ in range(self.n_iter):
            Y = A.dot(A.T.dot(Y))

        Q, _ = np.linalg.qr(Y, mode="reduced")
        B = A.T.dot(Q).T
        U_hat, S, VT = np.linalg.svd(B, full_matrices=False)

        U = Q.dot(U_hat)
        U = U[:, :k].astype(np.float32)
        S = S[:k].astype(np.float32)
        VT = VT[:k, :].astype(np.float32)
        return U, S, VT

    def __call__(self, dataset: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        print("Running spectral initialization by Randomized SVD...")
        try:
            R = dataset.trnMat.tocsr().astype(np.float32)
            user_deg = np.asarray(R.sum(axis=1)).flatten().astype(np.float32)
            item_deg = np.asarray(R.sum(axis=0)).flatten().astype(np.float32)
            user_inv = np.power(user_deg + 1e-10, -0.5)
            item_inv = np.power(item_deg + 1e-10, -0.5)
            norm_R = sp.diags(user_inv).dot(R).dot(sp.diags(item_inv)).astype(np.float32)

            min_dim = min(norm_R.shape[0], norm_R.shape[1])
            k = min(self.dim, max(1, min_dim - 1))
            if k < 1:
                raise RuntimeError("Matrix is too small for Randomized SVD initialization.")

            U, S, VT = self._randomized_svd_sparse(norm_R, k=k)
            sqrt_S = np.sqrt(S).astype(np.float32)
            user_feat = U.astype(np.float32) * sqrt_S.reshape(1, -1)
            item_feat = VT.T.astype(np.float32) * sqrt_S.reshape(1, -1)

            if k < self.dim:
                rng = np.random.default_rng(self.seed)
                user_pad = rng.normal(0, 0.01, size=(dataset.n_users, self.dim - k)).astype(np.float32)
                item_pad = rng.normal(0, 0.01, size=(dataset.n_items, self.dim - k)).astype(np.float32)
                user_feat = np.concatenate([user_feat, user_pad], axis=1)
                item_feat = np.concatenate([item_feat, item_pad], axis=1)

            user_feat = np.nan_to_num(user_feat, nan=0.0, posinf=0.0, neginf=0.0)
            item_feat = np.nan_to_num(item_feat, nan=0.0, posinf=0.0, neginf=0.0)
            user_feat = torch.FloatTensor(user_feat).to(device)
            item_feat = torch.FloatTensor(item_feat).to(device)
            return user_feat, item_feat
        except Exception as e:
            print(f"Randomized SVD initialization failed: {repr(e)}")
            print("Fallback: random normalized initialization.")
            user_feat = torch.randn(dataset.n_users, self.dim, device=device)
            item_feat = torch.randn(dataset.n_items, self.dim, device=device)
            user_feat = F.normalize(user_feat, dim=1)
            item_feat = F.normalize(item_feat, dim=1)
            return user_feat, item_feat
