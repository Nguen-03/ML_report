"""Kiểm định chéo và tìm siêu tham số (chỉ dùng tập train)."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .evaluation import metrics_at, rank_metrics, theory_threshold
from .models import N_ITER, build_pipeline, search_space
from .utils import C_FN, C_FP, SEED


def make_cv(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def cv_report(pipe, X, y, skf, w=1.0, r=C_FN / C_FP):
    """Kiểm định chéo: trả về (trung bình, độ lệch chuẩn ddof=1) của các chỉ số trên 5 fold,
    kèm PR-AUC trên phần huấn luyện của fold để đo chênh lệch quá khớp."""
    rows = []
    t = theory_threshold(w, r)
    for tr, te in skf.split(X, y):
        m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        p_te = m.predict_proba(X.iloc[te])[:, 1]
        p_tr = m.predict_proba(X.iloc[tr])[:, 1]
        rk = rank_metrics(y.iloc[te], p_te, w)
        mt = metrics_at(y.iloc[te], p_te, t, r, 1.0)
        rows.append(dict(pr_auc=rk["pr_auc"], roc_auc=rk["roc_auc"], brier=rk["brier"],
                         f1_theory=mt["f1"], cost_theory=mt["cost_per_customer"],
                         pr_auc_trainfold=average_precision_score(y.iloc[tr], p_tr)))
    d = pd.DataFrame(rows)
    return d.mean(), d.std(ddof=1)


def tune(name, mode, X, y, skf, n0, n1, fast=False):
    """RandomizedSearchCV 5 fold, tiêu chí PR-AUC; cấu hình tốt nhất được huấn luyện lại trên toàn bộ train."""
    pipe = build_pipeline(name, mode, list(X.columns), n0, n1)
    n_iter = 2 if fast else N_ITER[name]
    search = RandomizedSearchCV(pipe, search_space(name), n_iter=n_iter, scoring="average_precision",
                                cv=skf, refit=True, random_state=SEED,
                                n_jobs=-1 if name == "LR" else 1)   # RF/GB đã song song hoá bên trong
    search.fit(X, y)
    return search
