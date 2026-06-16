# -*- coding: utf-8 -*-
"""Data loading, graph construction, and negative sampling."""

import os
import pickle
from typing import Any, Tuple

import numpy as np
import scipy.sparse as sp
import torch


class DataHelper:
    def __init__(self, config: Any):
        self.config = config
        self.device = config.device
        print(f"Loading data from {config.dataset_dir} on {self.device}...")

        trn_path = os.path.join(config.dataset_dir, "trnMat.pkl")
        tst_path = os.path.join(config.dataset_dir, "tstMat.pkl")

        try:
            self.trnMat = pickle.load(open(trn_path, "rb"))
            self.tstMat = pickle.load(open(tst_path, "rb"))
        except Exception as e:
            if not config.demo_if_missing:
                raise RuntimeError(
                    f"Failed to load dataset from {config.dataset_dir}. "
                    f"Expected trnMat.pkl and tstMat.pkl. Original error: {repr(e)}"
                )
            print("Demo Mode: dataset files not found. Generating random demo data.")
            self.trnMat = sp.rand(10000, 10000, density=0.005, format="csr", random_state=2025)
            self.trnMat.data[:] = 1.0
            self.tstMat = sp.rand(10000, 10000, density=0.001, format="csr", random_state=2026)
            self.tstMat.data[:] = 1.0

        self.trnMat = self.trnMat.tocsr().astype(np.float32)
        self.tstMat = self.tstMat.tocsr().astype(np.float32)
        self.n_users, self.n_items = self.trnMat.shape
        self.n_nodes = self.n_users + self.n_items

        print(
            f"Dataset loaded: users={self.n_users}, items={self.n_items}, "
            f"train_interactions={self.trnMat.nnz}, test_interactions={self.tstMat.nnz}"
        )

        self.train_user_pos = [set(self.trnMat[u].indices.tolist()) for u in range(self.n_users)]

        self.adj_indices_np, self.adj_values_np, self.adj_shape = self._build_graph_components(self.trnMat)
        self.adj_indices = torch.LongTensor(self.adj_indices_np).to(self.device)
        self.adj_values = torch.FloatTensor(self.adj_values_np).to(self.device)
        self.graph = self._make_sparse_tensor(self.adj_indices, self.adj_values, self.adj_shape)
        self.interaction_sparse = self._make_user_item_sparse_tensor(self.trnMat)

        self.user_deg_np = np.asarray(self.trnMat.sum(axis=1)).flatten().astype(np.float32)
        self.item_deg_np = np.asarray(self.trnMat.sum(axis=0)).flatten().astype(np.float32)
        full_deg = np.concatenate([self.user_deg_np, self.item_deg_np])
        log_deg = np.log1p(full_deg)
        if log_deg.max() > 0:
            log_deg = log_deg / (log_deg.max() + 1e-10)
        self.norm_log_degree = torch.FloatTensor(log_deg).to(self.device)

    def sample_negatives(self, batch_users: torch.Tensor, neg_k: int, device: torch.device) -> torch.Tensor:
        if not self.config.exact_negative_sampling:
            return torch.randint(
                low=0,
                high=self.n_items,
                size=(batch_users.numel(), neg_k),
                device=device,
                dtype=torch.long,
            )

        users_np = batch_users.detach().cpu().numpy()
        neg_np = np.random.randint(0, self.n_items, size=(len(users_np), neg_k), dtype=np.int64)
        for row_idx, u in enumerate(users_np):
            pos_set = self.train_user_pos[int(u)]
            if len(pos_set) == 0:
                continue
            for col_idx in range(neg_k):
                tries = 0
                while int(neg_np[row_idx, col_idx]) in pos_set and tries < 100:
                    neg_np[row_idx, col_idx] = np.random.randint(0, self.n_items)
                    tries += 1
        return torch.LongTensor(neg_np).to(device)

    def _build_graph_components(self, mat: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        row, col = mat.nonzero()
        col_shifted = col + self.n_users
        data = np.ones(len(row), dtype=np.float32)

        full_row = np.concatenate([row, col_shifted])
        full_col = np.concatenate([col_shifted, row])
        full_data = np.concatenate([data, data]).astype(np.float32)
        indices = np.vstack([full_row, full_col]).astype(np.int64)

        return indices, full_data, (self.n_nodes, self.n_nodes)

    def _make_sparse_tensor(self, indices: torch.Tensor, values: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        adj = torch.sparse_coo_tensor(indices, values, shape).coalesce()
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)

        row = adj.indices()[0]
        col = adj.indices()[1]
        val = adj.values()
        norm_val = val * deg_inv_sqrt[row] * deg_inv_sqrt[col]

        norm_adj = torch.sparse_coo_tensor(adj.indices(), norm_val, shape).coalesce()
        return norm_adj

    def _make_user_item_sparse_tensor(self, mat: sp.csr_matrix) -> torch.Tensor:
        coo = mat.tocoo().astype(np.float32)
        idx = torch.LongTensor(np.vstack([coo.row, coo.col]))
        val = torch.FloatTensor(np.ones_like(coo.data, dtype=np.float32))
        tensor = torch.sparse_coo_tensor(idx, val, (self.n_users, self.n_items)).coalesce().to(self.device)
        return tensor
