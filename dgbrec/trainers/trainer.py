# -*- coding: utf-8 -*-
"""Training loop for DGBRec single-run experiments."""

from typing import Any, Dict
import copy
import os
import time

import numpy as np
import torch
import torch.optim as optim

from ..evaluation import test_fast
from ..models import DGBRec
from ..utils import append_csv, ensure_dir, set_seed, write_json


def run_training(config: Any, dataset: Any, run_name: str) -> Dict[str, Any]:
    set_seed(config.seed)
    ensure_dir(config.result_dir)
    run_dir = os.path.join(config.result_dir, run_name)
    ensure_dir(run_dir)
    write_json(os.path.join(run_dir, "config.json"), config.to_dict())

    model = DGBRec(dataset=dataset, config=config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.scheduler_eta_min)

    # train_coo = dataset.trnMat.tocoo()
    # train_u_tensor = torch.LongTensor(train_coo.row.astype(np.int64)).to(config.device)
    # train_v_tensor = torch.LongTensor(train_coo.col.astype(np.int64)).to(config.device)
    # num_samples = len(train_u_tensor)
    train_coo = dataset.trnMat.tocoo()

    train_u_np = train_coo.row.astype(np.int64)
    train_v_np = train_coo.col.astype(np.int64)
    num_samples = len(train_u_np)

    print("========== Train COO Check ==========")
    print("train_coo.shape:", train_coo.shape)
    print("dataset.n_users:", dataset.n_users)
    print("dataset.n_items:", dataset.n_items)
    print("row min/max:", train_u_np.min(), train_u_np.max())
    print("col min/max:", train_v_np.min(), train_v_np.max())
    print("nnz:", num_samples)

    if train_u_np.min() < 0 or train_u_np.max() >= dataset.n_users:
        raise ValueError(
            f"Train user index out of range: "
            f"valid=[0,{dataset.n_users - 1}], "
            f"got min={train_u_np.min()}, max={train_u_np.max()}"
        )

    if train_v_np.min() < 0 or train_v_np.max() >= dataset.n_items:
        raise ValueError(
            f"Train item index out of range: "
            f"valid=[0,{dataset.n_items - 1}], "
            f"got min={train_v_np.min()}, max={train_v_np.max()}"
        )

    best_primary = -1.0
    best_metrics: Dict[str, float] = {}
    best_epoch = 0
    patience_cnt = 0
    train_log_path = os.path.join(run_dir, "training_log.csv")
    fieldnames = [
        "run_name", "seed", "epoch", "loss", "gb_weight", "tau_h", "lr",
    ]
    for k in config.topks:
        fieldnames.extend([f"recall@{k}", f"ndcg@{k}"])
    fieldnames.extend(["best_primary", "best_epoch"])

    start_time = time.time()
    for epoch in range(config.epochs):
        decay_ratio = epoch / max(1, config.epochs - 1)
        current_tau_h = max(
            config.eval_tau_h,
            config.tau_h * (1.0 - decay_ratio) + config.eval_tau_h * decay_ratio,
        )
        gb_weight = min(1.0, float(epoch + 1) / float(max(1, config.gb_warmup_epochs)))

        model.train()
        # indices = torch.randperm(num_samples, device=config.device)
        # train_u_shuffled = train_u_tensor[indices]
        # train_v_shuffled = train_v_tensor[indices]
        # num_batches = (num_samples + config.batch_size - 1) // config.batch_size
        indices_np = np.random.permutation(num_samples)
        num_batches = (num_samples + config.batch_size - 1) // config.batch_size
        total_loss = 0.0
        valid_batch_count = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * config.batch_size
            end_idx = min((batch_idx + 1) * config.batch_size, num_samples)
            # batch_u = train_u_shuffled[start_idx:end_idx]
            # batch_pos = train_v_shuffled[start_idx:end_idx]
            batch_indices_np = indices_np[start_idx:end_idx]

            batch_u_np = train_u_np[batch_indices_np]
            batch_pos_np = train_v_np[batch_indices_np]

            # CPU-side safety check before moving to CUDA
            bu_min, bu_max = int(batch_u_np.min()), int(batch_u_np.max())
            bp_min, bp_max = int(batch_pos_np.min()), int(batch_pos_np.max())

            if bu_min < 0 or bu_max >= dataset.n_users:
                raise ValueError(
                    f"batch_u out of range at epoch={epoch + 1}, batch={batch_idx}: "
                    f"valid=[0,{dataset.n_users - 1}], got min={bu_min}, max={bu_max}"
                )

            if bp_min < 0 or bp_max >= dataset.n_items:
                raise ValueError(
                    f"batch_pos out of range at epoch={epoch + 1}, batch={batch_idx}: "
                    f"valid=[0,{dataset.n_items - 1}], got min={bp_min}, max={bp_max}"
                )

            batch_u = torch.from_numpy(batch_u_np).long().to(config.device, non_blocking=True)
            batch_pos = torch.from_numpy(batch_pos_np).long().to(config.device, non_blocking=True)
            if batch_u.numel() < max(1, config.batch_size // 2):
                continue
            batch_neg = dataset.sample_negatives(batch_users=batch_u, neg_k=config.neg_k, device=config.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.calculate_loss(
                batch_users=batch_u,
                batch_pos_items=batch_pos,
                batch_neg_items=batch_neg,
                current_tau_h=current_tau_h,
                gb_weight=gb_weight,
            )
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN or Inf loss detected in {run_name} at epoch {epoch + 1}.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item())
            valid_batch_count += 1

        scheduler.step()

        if (epoch + 1) % config.eval_interval == 0:
            metrics = test_fast(model=model, dataset=dataset, topks=config.topks)
            primary_value = metrics.get(config.primary_metric, metrics.get("ndcg@20", 0.0))
            avg_loss = total_loss / max(1, valid_batch_count)
            lr_now = optimizer.param_groups[0]["lr"]
            log_row = {
                "run_name": run_name,
                "seed": config.seed,
                "epoch": epoch + 1,
                "loss": avg_loss,
                "gb_weight": gb_weight,
                "tau_h": current_tau_h,
                "lr": lr_now,
            }
            log_row.update(metrics)

            if primary_value > best_primary:
                best_primary = primary_value
                best_metrics = copy.deepcopy(metrics)
                best_epoch = epoch + 1
                patience_cnt = 0
                if config.save_best_model:
                    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            else:
                patience_cnt += 1

            log_row["best_primary"] = best_primary
            log_row["best_epoch"] = best_epoch
            append_csv(train_log_path, log_row, fieldnames=fieldnames)

            metric_str = " | ".join([f"{k} {v:.4f}" for k, v in metrics.items()])
            print(
                f"[{run_name}] Epoch {epoch + 1:03d} | Loss {avg_loss:.4f} | {metric_str} | "
                f"Best {best_primary:.4f}@{best_epoch} | tau_h {current_tau_h:.4f} | gb_w {gb_weight:.3f}"
            )

            if patience_cnt >= config.patience:
                print(f"[{run_name}] Early stopping at epoch {epoch + 1}.")
                break

    total_time = time.time() - start_time

    result = {
        "run_name": run_name,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_primary_metric": config.primary_metric,
        "best_primary_value": best_primary,
        "time_sec": total_time,
    }
    result.update(best_metrics)

    for key in [
        "emb_dim", "batch_size", "l_r", "decay", "n_layers", "n_balls_u", "n_balls_i",
        "tau_h", "topm_membership", "rho_c", "rho_r", "detach_ball_update",
        "radius_init_scale", "lambda_cov", "lambda_rad", "lambda_purity", "lambda_mass",
        "eta_struct", "beta_quality", "neg_k",
    ]:
        result[key] = getattr(config, key, None)

    write_json(os.path.join(run_dir, "result.json"), result)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
