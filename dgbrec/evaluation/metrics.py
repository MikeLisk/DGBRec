# -*- coding: utf-8 -*-
"""Ranking evaluation metrics for recommendation."""

from typing import Any, Dict, List

import numpy as np
import torch


def test_fast(model: Any, dataset: Any, topks: List[int]) -> Dict[str, float]:
    model.eval()
    max_k = max(topks)
    with torch.no_grad():
        user_emb, item_emb = model.forward(current_tau_h=model.config.eval_tau_h, return_aux=False)
        user_emb = user_emb.detach().cpu()
        item_emb = item_emb.detach().cpu()

        test_dict: Dict[int, List[int]] = {}
        row, col = dataset.tstMat.nonzero()
        for r, c in zip(row, col):
            test_dict.setdefault(int(r), []).append(int(c))
        test_users = list(test_dict.keys())
        if len(test_users) == 0:
            result = {}
            for k in topks:
                result[f"recall@{k}"] = 0.0
                result[f"ndcg@{k}"] = 0.0
            return result

        batch_size = 2048
        metrics_hits = {k: [] for k in topks}
        metrics_ndcgs = {k: [] for k in topks}
        trn_csr = dataset.trnMat.tocsr()

        for start in range(0, len(test_users), batch_size):
            end = min(start + batch_size, len(test_users))
            batch_u_ids = test_users[start:end]
            batch_user_emb = user_emb[batch_u_ids]
            batch_scores = torch.mm(batch_user_emb, item_emb.t())
            for idx, user_id in enumerate(batch_u_ids):
                pos_items = trn_csr[user_id].indices
                batch_scores[idx, pos_items] = -1e9
            _, indices = torch.topk(batch_scores, k=max_k, dim=1)
            indices = indices.numpy()

            for idx, user_id in enumerate(batch_u_ids):
                gt_items = test_dict[user_id]
                gt_len = len(gt_items)
                for k in topks:
                    rec_items = indices[idx, :k]
                    hit_mask = np.isin(rec_items, gt_items)
                    hit_cnt = int(hit_mask.sum())
                    metrics_hits[k].append(hit_cnt / gt_len)
                    if hit_cnt > 0:
                        hit_pos = np.where(hit_mask)[0]
                        dcg = np.sum(1.0 / np.log2(hit_pos + 2))
                        idcg = np.sum(1.0 / np.log2(np.arange(min(gt_len, k)) + 2))
                        metrics_ndcgs[k].append(float(dcg / idcg))
                    else:
                        metrics_ndcgs[k].append(0.0)

    result = {}
    for k in topks:
        result[f"recall@{k}"] = float(np.mean(metrics_hits[k]))
        result[f"ndcg@{k}"] = float(np.mean(metrics_ndcgs[k]))
    return result
