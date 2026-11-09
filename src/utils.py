"""Hằng số dùng chung và các hàm tiện ích (đọc/ghi, mã băm, phiên bản thư viện)."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

SEED = 42                      # hạt giống cho mọi nguồn ngẫu nhiên
C_FN, C_FP = 5.0, 1.0          # ma trận chi phí cơ sở (r = 5)
TARGET = "default_next_month"   # tên cột nhãn như trong notebook/EDA.ipynb

CAT = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
MONEY = ["LIMIT_BAL"] + BILL_COLS + PAYAMT_COLS + ["bill_mean", "pay_mean", "limit_per_age"]

THRESH_GRID = np.round(np.arange(0.02, 0.98 + 1e-9, 0.005), 3)   # lưới ngưỡng 0,02 -> 0,98
R_GRID = [1, 2, 3, 5, 8, 10, 15, 20]                              # lưới tỉ lệ chi phí

MODES = ["none", "balanced", "cost"]
ALGOS = ["LR", "RF", "GB"]


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def update_run_log(path, key, value):
    """Ghi thêm một mục vào run_log.json (giữ nguyên các mục đã có)."""
    path = Path(path)
    log = load_json(path) if path.exists() else {}
    log[key] = value
    save_json(log, path)


def sha256_of_list(values):
    return hashlib.sha256(json.dumps(sorted(int(v) for v in values)).encode()).hexdigest()


def versions():
    import joblib, matplotlib, pandas, scipy, sklearn
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "scikit-learn": sklearn.__version__,
        "pandas": pandas.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
    }
