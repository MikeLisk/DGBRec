# -*- coding: utf-8 -*-
"""Entry point for DGBRec single-run training.

All default hyperparameters are defined in this file so that the experiment
configuration is easy to read, modify, and reproduce from one place.
"""

import argparse
import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Environment settings should be applied before heavy numerical work starts.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from dgbrec.data import DataHelper
from dgbrec.trainers import run_training
from dgbrec.utils import ensure_dir, parse_int_list, save_csv


@dataclass
class DGBRecConfig:
    """Best default configuration for the DGBRec single-run experiment.

    This project intentionally keeps the default experiment parameters here in
    main.py. The remaining modules receive this config object but do not define
    or override experiment settings.
    """

    # Experiment settings
    seed: int = 2025
    seeds: List[int] = field(default_factory=lambda: [2025])
    dataset_dir: str = "./Datasets/sparse_amazon"
    result_dir: str = "./results_dgbrec"
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cudnn_benchmark: bool = True

    # Data and sampling settings
    exact_negative_sampling: bool = False
    demo_if_missing: bool = False

    # Training settings
    epochs: int = 400
    patience: int = 30
    eval_interval: int = 5
    batch_size: int = 4096
    lr: float = 0.004254953399591034
    l_r: float = 0.004254953399591034
    weight_decay: float = 0.0006906257175600949
    decay: float = 0.0006906257175600949
    neg_k: int = 16
    topks: List[int] = field(default_factory=lambda: [20, 40])
    primary_metric: str = "ndcg@20"
    save_best_model: bool = False
    grad_clip_norm: float = 5.0
    scheduler_eta_min: float = 1e-5

    # Randomized SVD spectral initialization settings
    random_svd_oversampling: int = 16
    random_svd_n_iter: int = 3
    random_svd_seed: int = 2025

    # DGBRec model settings
    emb_dim: int = 128
    n_layers: int = 2
    n_balls_u: int = 768
    n_balls_i: int = 512
    tau_h: float = 0.3795539116630165
    eval_tau_h: float = 0.1
    topm_membership: int = 4
    rho_c: float = 0.840658040985111
    rho_r: float = 0.6044918634787524
    detach_ball_update: bool = False
    radius_init_scale: float = 1.8453142670816267
    eta_struct: float = 0.20
    beta_quality: float = 1.00

    # Granular-ball loss and regularization settings
    kmeans_iters: int = 8
    radius_eps: float = 1e-6
    radius_min: float = 1e-4
    radius_max: float = 10.0
    coverage_margin: float = 1.0
    q_floor: float = 0.05
    gb_warmup_epochs: int = 50
    lambda_cov: float = 0.002686455772166749
    lambda_rad: float = 0.00010491319031455553
    lambda_purity: float = 0.005492267676420222
    lambda_mass: float = 9.761976299435802e-05

    def clone_for_seed(self, seed: int) -> "DGBRecConfig":
        cfg = copy.deepcopy(self)
        cfg.seed = int(seed)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, torch.device):
                result[key] = str(value)
            elif isinstance(value, (int, float, str, bool, type(None))):
                result[key] = value
            elif isinstance(value, list):
                result[key] = value
            else:
                result[key] = str(value)
        return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DGBRec single run with best default configuration.")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Directory containing trnMat.pkl and tstMat.pkl.")
    parser.add_argument("--result_dir", type=str, default=None, help="Directory for saving logs and results.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience counted by evaluations.")
    parser.add_argument("--eval_interval", type=int, default=None, help="Evaluate every N epochs.")
    parser.add_argument("--topks", type=str, default=None, help="Comma-separated Top-K values, e.g., 20,40.")
    parser.add_argument("--seed", type=int, default=None, help="Single random seed. Overrides the first seed.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g., 2025,2026,2027.")
    parser.add_argument("--device", type=str, default=None, help="Device name, e.g., cuda, cuda:0, or cpu. Default: auto.")
    parser.add_argument("--exact_negative_sampling", action="store_true", help="Avoid sampling known positive items as negatives.")
    parser.add_argument("--save_best_model", action="store_true", help="Save best_model.pt when validation improves.")
    parser.add_argument("--demo_if_missing", action="store_true", help="Generate random demo data if dataset files are missing.")
    return parser


def build_config_from_args(args: argparse.Namespace) -> DGBRecConfig:
    cfg = DGBRecConfig()

    if args.dataset_dir is not None:
        cfg.dataset_dir = args.dataset_dir
    if args.result_dir is not None:
        cfg.result_dir = args.result_dir
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.patience is not None:
        cfg.patience = int(args.patience)
    if args.eval_interval is not None:
        cfg.eval_interval = int(args.eval_interval)
    if args.topks is not None:
        cfg.topks = parse_int_list(args.topks) or cfg.topks
        if 20 not in cfg.topks:
            cfg.primary_metric = f"ndcg@{cfg.topks[0]}"
    if args.seed is not None:
        cfg.seed = int(args.seed)
        cfg.seeds = [int(args.seed)]
    if args.seeds is not None:
        cfg.seeds = parse_int_list(args.seeds) or cfg.seeds
        cfg.seed = cfg.seeds[0]
    if args.device is not None:
        cfg.device = torch.device(args.device)
    if args.exact_negative_sampling:
        cfg.exact_negative_sampling = True
    if args.save_best_model:
        cfg.save_best_model = True
    if args.demo_if_missing:
        cfg.demo_if_missing = True

    # Keep synonymous names consistent for modules and saved summaries.
    cfg.l_r = cfg.lr
    cfg.decay = cfg.weight_decay
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    return cfg


def run_single(config: DGBRecConfig) -> None:
    ensure_dir(config.result_dir)
    all_results: List[Dict[str, Any]] = []

    for seed in config.seeds:
        cfg = config.clone_for_seed(seed)
        dataset = DataHelper(cfg)
        run_name = f"DGBRec_seed{seed}"
        result = run_training(cfg, dataset, run_name=run_name)
        all_results.append(result)

    summary_path = os.path.join(config.result_dir, "single_summary.csv")
    save_csv(summary_path, all_results)

    print("\n================ DGBRec Single Run Summary ================")
    for row in all_results:
        print(row)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = build_config_from_args(args)

    print("================ DGBRec Single Run ================")
    print(f"Dataset directory: {config.dataset_dir}")
    print(f"Result directory: {config.result_dir}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.epochs}")
    print(f"Patience: {config.patience}")
    print(f"Eval interval: {config.eval_interval}")
    print(f"TopKs: {config.topks}")
    print(f"Seeds: {config.seeds}")

    run_single(config)


if __name__ == "__main__":
    main()
