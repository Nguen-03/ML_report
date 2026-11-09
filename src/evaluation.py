"""Chỉ số, chọn ngưỡng, chi phí, bootstrap, phân tích theo nhóm và vẽ hình."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

from .utils import C_FN, C_FP, SEED, THRESH_GRID


# ---------------------------------------------------------------- ngưỡng và chi phí
def theory_threshold(w, r=C_FN / C_FP):
    """Ngưỡng tối ưu lý thuyết trên xác suất của mô hình huấn luyện với trọng số w: w/(w + r)."""
    return w / (w + r)


def unweight(p, w):
    """Đưa xác suất của mô hình có trọng số w về thang xác suất gốc (công thức 2.2 đảo)."""
    p = np.asarray(p, dtype=float)
    return p / (p + w * (1.0 - p) + 1e-12)


def metrics_at(y, p, t, c_fn=C_FN, c_fp=C_FP):
    y = np.asarray(y)
    pred = np.asarray(p) >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    cost = c_fn * fn + c_fp * fp
    return dict(threshold=float(t), tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                f1=f1, specificity=spec, youden=recall + spec - 1,
                total_cost=float(cost), cost_per_customer=float(cost / len(y)))


def f1_at(y, p, t):
    return metrics_at(y, p, t)["f1"]


def cost_scorer(w=1.0, r=C_FN / C_FP):
    """Scorer tuỳ biến (estimator, X, y) -> -chi phí trung bình tại ngưỡng lý thuyết."""
    def scorer(est, X, y):
        p = est.predict_proba(X)[:, 1]
        return -metrics_at(y, p, theory_threshold(w, r), r, 1.0)["cost_per_customer"]
    return scorer


def sweep(y, p, grid=THRESH_GRID, c_fn=C_FN, c_fp=C_FP):
    """Quét ngưỡng (vector hoá). Trả về DataFrame một dòng cho mỗi ngưỡng."""
    y = np.asarray(y); p = np.asarray(p)
    pred = p[:, None] >= np.asarray(grid)[None, :]
    pos, neg = (y == 1)[:, None], (y == 0)[:, None]
    tp = (pred & pos).sum(0); fp = (pred & neg).sum(0)
    fn = (~pred & pos).sum(0); tn = (~pred & neg).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        spec = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    cost = c_fn * fn + c_fp * fp
    return pd.DataFrame(dict(threshold=grid, precision=precision, recall=recall, f1=f1,
                             youden=recall + spec - 1, total_cost=cost, cost_per_customer=cost / len(y)))


def best_threshold(sw, criterion):
    """criterion: 'f1' | 'youden' | 'cost'. Trả về ngưỡng tối ưu từ bảng quét."""
    if criterion == "cost":
        return float(sw.loc[sw["total_cost"].idxmin(), "threshold"])
    return float(sw.loc[sw[criterion].idxmax(), "threshold"])


def naive_costs(y, c_fn=C_FN, c_fp=C_FP):
    """Hai chiến lược ngây thơ: chấp thuận tất cả (không cảnh báo) và từ chối tất cả."""
    y = np.asarray(y); n1 = int(y.sum()); n0 = int(len(y) - n1)
    accept_all, reject_all = c_fn * n1, c_fp * n0
    prec = n1 / len(y)
    return dict(n=len(y), n1=n1, n0=n0, accept_all=accept_all, reject_all=reject_all,
                accept_all_per_customer=accept_all / len(y), reject_all_per_customer=reject_all / len(y),
                reject_all_precision=prec, reject_all_f1=2 * prec / (1 + prec),
                best_naive=min(accept_all, reject_all))


def saving(total_cost, y, c_fn=C_FN, c_fp=C_FP):
    """Công thức (2.6): S = 1 - TC / min(CFN*N1, CFP*N0)."""
    return 1.0 - total_cost / naive_costs(y, c_fn, c_fp)["best_naive"]


def rank_metrics(y, p, w=1.0):
    return dict(pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
                brier=brier_score_loss(y, unweight(p, w)))


def bootstrap_cost_ci(y, p, t, n_boot=1000, seed=SEED, c_fn=C_FN, c_fp=C_FP):
    """Khoảng tin cậy 95% (phân vị 2,5% và 97,5%) của chi phí trung bình/khách."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = metrics_at(y[i], p[i], t, c_fn, c_fp)["cost_per_customer"]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- độ nhạy với r
def sensitivity(y, p_model, w, r_grid, grid=THRESH_GRID):
    """Với mỗi r chọn lại ngưỡng trên validation. Mọi phép tính trên thang xác suất gốc p0."""
    p0 = unweight(p_model, w)
    rows = []
    for r in r_grid:
        sw = sweep(y, p0, grid, c_fn=r, c_fp=1.0)
        t0 = best_threshold(sw, "cost")
        m = metrics_at(y, p0, t0, r, 1.0)
        t_model = w * t0 / (w * t0 + 1 - t0)
        rows.append(dict(r=r, theory_threshold=1 / (1 + r), empirical_threshold=t0,
                         empirical_threshold_model_scale=t_model, recall=m["recall"],
                         precision=m["precision"], cost_per_customer=m["cost_per_customer"]))
    return pd.DataFrame(rows)


def regret_matrix(y, p_model, w, r_grid, grid=THRESH_GRID):
    """Công thức (2.7): hàng = giả định r_a, cột = chi phí thật r_t."""
    p0 = unweight(p_model, w)
    t_opt = {r: best_threshold(sweep(y, p0, grid, c_fn=r), "cost") for r in r_grid}
    out = pd.DataFrame(index=r_grid, columns=r_grid, dtype=float)
    for ra in r_grid:
        for rt in r_grid:
            tc_a = metrics_at(y, p0, t_opt[ra], rt, 1.0)["total_cost"]
            tc_t = metrics_at(y, p0, t_opt[rt], rt, 1.0)["total_cost"]
            out.loc[ra, rt] = (tc_a - tc_t) / tc_t if tc_t else np.nan
    out.index.name = "r_assumed"; out.columns.name = "r_true"
    return out


# ---------------------------------------------------------------- phân tích lỗi theo nhóm
def make_groups(X, X_train):
    q = np.quantile(X_train["LIMIT_BAL"], [0.25, 0.5, 0.75])
    edu = {1: "Sau đại học", 2: "Đại học", 3: "Trung học", 4: "Khác"}
    g = {}
    g["PAY_1"] = pd.cut(X["PAY_1"], [-10, -1, 0, 1, 2, 20], labels=["≤ −1", "0", "1", "2", "≥ 3"])
    g["LIMIT_BAL"] = pd.cut(X["LIMIT_BAL"], [-np.inf, q[0], q[1], q[2], np.inf],
                            labels=["Tứ phân vị 1", "Tứ phân vị 2", "Tứ phân vị 3", "Tứ phân vị 4"])
    g["AGE"] = pd.cut(X["AGE"], [0, 25, 35, 45, 55, 200], labels=["≤ 25", "26–35", "36–45", "46–55", "> 55"])
    g["EDUCATION"] = pd.Categorical(X["EDUCATION"].map(edu), categories=list(edu.values()), ordered=True)
    g["n_late"] = pd.cut(X["n_late"], [-1, 0, 1, 2, 10], labels=["0", "1", "2", "≥ 3"])
    return {k: pd.Series(v, index=X.index) for k, v in g.items()}


def error_by_group(y, pred, groups, c_fn=C_FN, c_fp=C_FP):
    y = np.asarray(y); pred = np.asarray(pred).astype(bool)
    fn_all = (~pred & (y == 1)); fp_all = (pred & (y == 0))
    total_cost = c_fn * fn_all.sum() + c_fp * fp_all.sum()
    rows = []
    for gname, lab in groups.items():
        for lv in lab.cat.categories:
            m = (lab == lv).values
            n = int(m.sum())
            if n == 0:
                continue
            n1 = int((y[m] == 1).sum()); n0 = n - n1
            fn = int(fn_all[m].sum()); fp = int(fp_all[m].sum())
            rows.append({"Nhóm theo": gname, "Mức": str(lv), "Số khách": n,
                         "Tỉ lệ vỡ nợ": n1 / n, "FN rate": fn / n1 if n1 else np.nan,
                         "FP rate": fp / n0 if n0 else np.nan,
                         "Đóng góp vào tổng chi phí": (c_fn * fn + c_fp * fp) / total_cost if total_cost else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- hình
def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc_pr(y, p_dict, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in p_dict.items():
        fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
        ax[1].plot(rc, pr, label=f"{name} (AP={average_precision_score(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set(xlabel="FPR", ylabel="TPR", title="ROC (validation)")
    ax[1].axhline(np.mean(y), color="k", ls="--", lw=0.8)
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall (validation)")
    for a in ax: a.legend(fontsize=7); a.grid(alpha=0.3)
    _save(fig, path)


def plot_cm(m, title, path):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Dự đoán 0", "Dự đoán 1"],
           yticklabels=["Thực tế 0", "Thực tế 1"], title=title)
    _save(fig, path)


def plot_cost_curve(sweeps, naive, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, sw in sweeps.items():
        ax.plot(sw["threshold"], sw["total_cost"], label=name)
    ax.axhline(naive["reject_all"], color="gray", ls="--", label="Từ chối tất cả")
    ax.axhline(naive["accept_all"], color="gray", ls=":", label="Chấp thuận tất cả")
    ax.axvline(1 / (1 + C_FN / C_FP), color="r", ls="--", lw=0.8, label="Ngưỡng lý thuyết 1/(1+r)")
    ax.set(xlabel="Ngưỡng quyết định", ylabel="Tổng chi phí", title="Tổng chi phí theo ngưỡng (validation)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    _save(fig, path)


def plot_residual_calibration(y, p0, path):
    y = np.asarray(y)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    res = y - p0
    ax[0].hist(res[y == 0], bins=40, alpha=0.6, label="Không vỡ nợ"); ax[0].hist(res[y == 1], bins=40, alpha=0.6, label="Vỡ nợ")
    ax[0].set(xlabel="Phần dư y − p̂", ylabel="Số khách", title="Phân phối phần dư (test)"); ax[0].legend()
    frac, mean = calibration_curve(y, p0, n_bins=10, strategy="quantile")
    ax[1].plot(mean, frac, "o-", label="Mô hình"); ax[1].plot([0, 1], [0, 1], "k--", label="Hiệu chỉnh hoàn hảo")
    ax[1].set(xlabel="Xác suất dự đoán (thang gốc)", ylabel="Tỉ lệ vỡ nợ thực tế", title="Đường hiệu chỉnh (test)"); ax[1].legend()
    for a in ax: a.grid(alpha=0.3)
    _save(fig, path)


def plot_sensitivity(sens, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.plot(sens["r"], sens["theory_threshold"], "o--", label="Lý thuyết 1/(1+r)")
    ax.plot(sens["r"], sens["empirical_threshold"], "s-", label="Thực nghiệm (validation)")
    ax.set(xlabel="Tỉ lệ chi phí r = CFN/CFP", ylabel="Ngưỡng tối ưu (thang xác suất gốc)", title="Độ nhạy của ngưỡng theo r")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, path)


def plot_importance(imp, path, top=15):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"])
    ax.set(xlabel="Mức giảm PR-AUC khi xáo trộn", title="Permutation importance (validation)")
    _save(fig, path)
