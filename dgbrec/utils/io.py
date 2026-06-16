# -*- coding: utf-8 -*-
"""Utility functions for DGBRec experiments."""

import csv
import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def parse_int_list(text: str) -> List[int]:
    if text is None or str(text).strip() == "":
        return []
    return [int(x.strip()) for x in str(text).split(",") if x.strip() != ""]


def safe_float(x: Any) -> Any:
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64)):
        return int(x)
    if isinstance(x, torch.device):
        return str(x)
    return x


def write_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    clean = {k: safe_float(v) for k, v in data.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)


def append_csv(path: str, row: Dict[str, Any], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(os.path.dirname(path))
    if fieldnames is None:
        fieldnames = list(row.keys())
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: safe_float(row.get(k, "")) for k in fieldnames})


def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(os.path.dirname(path))
    if len(rows) == 0:
        return
    if fieldnames is None:
        keys = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: safe_float(row.get(k, "")) for k in fieldnames})
