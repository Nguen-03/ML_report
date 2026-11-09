"""Đọc dữ liệu thô, làm sạch, tạo đặc trưng phái sinh và chia tập train/val/test."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import (BILL_COLS, PAY_COLS, PAYAMT_COLS, SEED, TARGET,
                    ensure_dir, load_json, save_json, sha256_of_list)


# Tên cột như trong notebook/EDA.ipynb (dùng khi tệp có tiêu đề X1..X23, Y)
RENAME_MAP = {"default payment next month": TARGET, "default.payment.next.month": TARGET, "Y": TARGET,
              "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE", "X5": "AGE",
              "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3", "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
              **{f"X{11 + i}": f"BILL_AMT{i}" for i in range(1, 7)},
              **{f"X{17 + i}": f"PAY_AMT{i}" for i in range(1, 7)}}


def load_raw(path):
    """Đọc tệp UCI .xls/.xlsx (dòng 0 là tiêu đề X1..X23,Y nên dùng header=1) hoặc .csv."""
    p = Path(path)
    if p.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(p, header=1)
    else:
        df = pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    return df.rename(columns=RENAME_MAP).rename(columns={"PAY_0": "PAY_1"})


def clean(df):
    """Bỏ ID, loại dòng trùng, gộp mã không mô tả. Trả về (df_sạch, thông tin)."""
    info = {"n_raw": int(len(df)), "positives_raw": int(df[TARGET].sum())}
    df = df.drop(columns=[c for c in ["ID"] if c in df.columns])
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    info["n_duplicates_removed"] = int(n_before - len(df))
    # EDUCATION: 0, 5, 6 không có trong mô tả -> gộp vào 4 (Khác); MARRIAGE: 0 -> 3 (Khác)
    info["education_recoded"] = int(df["EDUCATION"].isin([0, 5, 6]).sum())
    info["marriage_recoded"] = int((df["MARRIAGE"] == 0).sum())
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    assert df.isna().sum().sum() == 0, "Dữ liệu còn giá trị thiếu"
    info["n_clean"] = int(len(df))
    info["positives_clean"] = int(df[TARGET].sum())
    info["default_rate_clean"] = float(df[TARGET].mean())
    return df, info


def load_clean(path):
    return clean(load_raw(path))


def add_features(df):
    """8 đặc trưng phái sinh; mỗi đặc trưng chỉ là hàm của từng dòng (không dùng thống kê toàn tập)."""
    df = df.copy()
    df["n_late"] = (df[PAY_COLS] > 0).sum(axis=1)
    df["max_delay"] = df[PAY_COLS].max(axis=1)
    df["util_1"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
    df["bill_mean"] = df[BILL_COLS].mean(axis=1)
    df["pay_mean"] = df[PAYAMT_COLS].mean(axis=1)
    df["util_mean"] = df["bill_mean"] / df["LIMIT_BAL"]
    df["pay_ratio"] = (df["pay_mean"] / df["bill_mean"].clip(lower=1)).clip(upper=10)
    df["limit_per_age"] = df["LIMIT_BAL"] / df["AGE"]
    return df


def make_split(df, split_path, seed=SEED):
    """Chia phân tầng 70/15/15. Nếu tệp chỉ số đã tồn tại thì đọc lại (không chia mới)."""
    split_path = Path(split_path)
    if split_path.exists():
        idx = load_json(split_path)
        all_idx = sorted(idx["train"] + idx["val"] + idx["test"])
        assert all_idx == list(range(len(df))), \
            "split_indices.json không khớp dữ liệu hiện tại; xoá tệp để chia lại"
        assert sha256_of_list(idx["test"]) == idx["sha256_test"], "Chỉ số test đã bị thay đổi"
        return idx
    ids = np.arange(len(df))
    y = df[TARGET].values
    # Cùng cách chia với notebook/EDA.ipynb (mục 8): 70% / 30%, rồi 30% chia đôi thành validation và test
    train, rest = train_test_split(ids, test_size=0.30, random_state=seed, stratify=y)
    val, test = train_test_split(rest, test_size=0.50, random_state=seed, stratify=y[rest])
    idx = {"seed": seed,
           "train": sorted(int(i) for i in train),
           "val": sorted(int(i) for i in val),
           "test": sorted(int(i) for i in test)}
    idx["sha256_test"] = sha256_of_list(idx["test"])
    save_json(idx, split_path)
    return idx


def get_xy(df, ids):
    sub = df.iloc[ids]
    return sub.drop(columns=[TARGET]), sub[TARGET].astype(int)


def save_processed(df, idx, folder, parts=("train", "val")):
    """Ghi data/processed/{train,val,test}_features.csv (đặc trưng phái sinh, chưa mã hoá one-hot).
    Hậu tố _features để không ghi đè train/val/test.csv (đã one-hot) do notebook tạo ra.
    Tập test chỉ được ghi ở giai đoạn final."""
    folder = ensure_dir(folder)
    for part in parts:
        df.iloc[idx[part]].to_csv(folder / f"{part}_features.csv", index=False)
