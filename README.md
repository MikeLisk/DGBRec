# DGBRec

DGBRec is a modular single-run implementation for recommendation experiments. This repository keeps the best default configuration for one standard training and evaluation pipeline.

The default experiment parameters are defined directly in `main.py` inside `DGBRecConfig`, as requested. The remaining modules only implement data processing, model components, evaluation, and training.

## Project structure

```text
DGBRec/
├── main.py                         # experiment parameters and single-run entry
├── requirements.txt
├── scripts/
│   └── run_dgbrec.sh
└── dgbrec/
    ├── data/
    │   └── data_helper.py          # data loading, graph construction, negative sampling
    ├── evaluation/
    │   └── metrics.py              # Recall@K and NDCG@K evaluation
    ├── models/
    │   ├── dgbrec.py               # DGBRec model framework
    │   ├── gates.py                # degree-aware fusion gate
    │   ├── granular_ball.py        # differentiable granular-ball module
    │   └── spectral.py             # randomized SVD initialization
    ├── trainers/
    │   └── trainer.py              # training loop and early stopping
    └── utils/
        └── io.py                   # seed, CSV, JSON, and filesystem utilities
```

## Expected dataset format

Place two pickled scipy sparse matrices in the dataset directory:

```text
Datasets/sparse_amazon/
├── trnMat.pkl
└── tstMat.pkl
```

Both matrices should have shape `[num_users, num_items]`. Rows represent users and columns represent items.

## Installation

```bash
pip install -r requirements.txt
```

## Run DGBRec with the best default configuration

```bash
python main.py --dataset_dir ./Datasets/sparse_amazon --result_dir ./results_dgbrec
```

You can also run:

```bash
bash scripts/run_dgbrec.sh
```

## Common command-line overrides

The full best configuration is in `main.py`. These command-line arguments only override common runtime settings:

```bash
python main.py  --dataset_dir ./Datasets/sparse_amazon --result_dir ./results_dgbrec
```

## Output files

```text
results_dgbrec/
├── DGBRec_seed2025/
│   ├── config.json
│   ├── training_log.csv
│   └── result.json
└── single_summary.csv
```
