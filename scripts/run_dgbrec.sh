#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --dataset_dir ./Datasets/sparse_amazon \
  --result_dir ./results_dgbrec \
  --epochs 400 \
  --patience 30 \
  --eval_interval 5 \
  --topks 20,40 \
  --seeds 2025
