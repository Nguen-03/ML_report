"""Toàn bộ quy trình thực nghiệm, từ dữ liệu thô đến bảng và hình của báo cáo.

  python run_experiment.py --stage dev     # chỉ dùng train + validation
  python run_experiment.py --stage final   # đánh giá MỘT LẦN trên test đã đóng băng
"""
import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from src import evaluation as ev
from src.data_loader import add_features, get_xy, load_clean, make_split, save_processed
from src.models import baselines, weight_value
from src.training import cv_report, make_cv, tune
from src.utils import (ALGOS, C_FN, C_FP, MODES, R_GRID, SEED, THRESH_GRID, ensure_dir, load_json,
                       save_json, update_run_log, versions)

warnings.filterwarnings("ignore")
CRITERIA = ("f1", "youden", "cost")


def _params(d):
    return {k.replace("clf__", ""): (v if not isinstance(v, (np.floating, np.integer)) else v.item())
            for k, v in d.items()}


def _paths(args):
    out = ensure_dir(args.out)
    split = Path(args.split)
    return out, ensure_dir(out / "tables"), ensure_dir(out / "figures"), ensure_dir(out / "models"), split


# ====================================================================== GIAI ĐOẠN DEV
def stage_dev(args):
    t_start = time.time()
    out, tables, figs, models_dir, split_path = _paths(args)
    df, info = load_clean(args.data)
    df = add_features(df)
    idx = make_split(df, split_path)
    X_tr, y_tr = get_xy(df, idx["train"])
    X_val, y_val = get_xy(df, idx["val"])
    save_processed(df, idx, args.processed, parts=("train", "val"))
    del df                                       # giai đoạn dev: tập test không được dùng
    n1 = int(y_tr.sum()); n0 = int(len(y_tr) - n1)
    skf = make_cv()
    print(f"[dev] train={len(X_tr)} val={len(X_val)} features={X_tr.shape[1]} w_balanced={n0 / n1:.3f}")

    # ---- B. mô hình cơ sở (Bảng 3.2)
    base_rows, base_val_p = [], {}
    for name, est in baselines().items():
        m_cv, s_cv = cv_report(est, X_tr, y_tr, skf, w=1.0)
        p = clone(est).fit(X_tr, y_tr).predict_proba(X_val)[:, 1]
        base_val_p[name] = p
        sw = ev.sweep(y_val, p); t = ev.best_threshold(sw, "cost"); m = ev.metrics_at(y_val, p, t)
        rk = ev.rank_metrics(y_val, p)
        base_rows.append(dict(model=name, pr_auc_cv=m_cv.pr_auc, pr_auc_cv_std=s_cv.pr_auc,
                              roc_auc_cv=m_cv.roc_auc, roc_auc_cv_std=s_cv.roc_auc,
                              pr_auc_val=rk["pr_auc"], roc_auc_val=rk["roc_auc"], recall=m["recall"],
                              precision=m["precision"], f1=m["f1"], total_cost=m["total_cost"],
                              cost_per_customer=m["cost_per_customer"]))
        print(f"[dev] baseline {name}: cost/khách={m['cost_per_customer']:.4f}")
    nv = ev.naive_costs(y_val)
    base_rows.append(dict(model="Chấp thuận tất cả", recall=0.0, total_cost=nv["accept_all"],
                          cost_per_customer=nv["accept_all_per_customer"]))
    base_rows.append(dict(model="Từ chối tất cả", recall=1.0, precision=nv["reject_all_precision"],
                          f1=nv["reject_all_f1"], total_cost=nv["reject_all"],
                          cost_per_customer=nv["reject_all_per_customer"]))
    pd.DataFrame(base_rows).to_csv(tables / "baselines.csv", index=False)

    # ---- 9 mô hình ứng viên: tìm siêu tham số, CV, chọn ngưỡng
    cands, cv_rows, thr_rows = {}, [], []
    for name in ALGOS:
        for mode in MODES:
            key = f"{name} – {mode}"
            t0 = time.time()
            search = tune(name, mode, X_tr, y_tr, skf, n0, n1, fast=args.fast)
            est = search.best_estimator_
            w = weight_value(mode, n0, n1)
            m_cv, s_cv = cv_report(est, X_tr, y_tr, skf, w)
            p_val = est.predict_proba(X_val)[:, 1]
            sw = ev.sweep(y_val, p_val)
            ths = {c: ev.best_threshold(sw, c) for c in CRITERIA}
            cands[key] = dict(est=est, w=w, p_val=p_val, sw=sw, thresholds=ths, name=name, mode=mode,
                              best_params=_params(search.best_params_))
            cv_rows.append(dict(model=key, algo=name, mode=mode, w=w, best_params=str(_params(search.best_params_)),
                                search_pr_auc=search.best_score_,
                                pr_auc_mean=m_cv.pr_auc, pr_auc_std=s_cv.pr_auc,
                                roc_auc_mean=m_cv.roc_auc, roc_auc_std=s_cv.roc_auc,
                                brier_mean=m_cv.brier, brier_std=s_cv.brier,
                                f1_theory_mean=m_cv.f1_theory, f1_theory_std=s_cv.f1_theory,
                                cost_theory_mean=m_cv.cost_theory, cost_theory_std=s_cv.cost_theory,
                                pr_auc_trainfold_mean=m_cv.pr_auc_trainfold,
                                pr_auc_gap=m_cv.pr_auc_trainfold - m_cv.pr_auc))
            for c in CRITERIA:
                m = ev.metrics_at(y_val, p_val, ths[c])
                thr_rows.append(dict(model=key, criterion=c, threshold=m["threshold"], recall=m["recall"],
                                     precision=m["precision"], f1=m["f1"], total_cost=m["total_cost"]))
            print(f"[dev] {key}: PR-AUC(cv)={m_cv.pr_auc:.4f}±{s_cv.pr_auc:.4f} "
                  f"thr_cost={ths['cost']:.3f} ({time.time() - t0:.0f}s)")
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(tables / "cv_results.csv", index=False)
    cv_df[["model", "pr_auc_trainfold_mean", "pr_auc_mean", "pr_auc_gap"]].to_csv(tables / "cv_gap.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(tables / "thresholds_val.csv", index=False)

    # ---- xếp hạng trên validation (Bảng 3.6) và chọn mô hình theo chi phí thấp nhất
    rank_rows = [dict(model=r["model"], pr_auc=r.get("pr_auc_val"), roc_auc=r.get("roc_auc_val"), recall=r.get("recall"),
                      precision=r.get("precision"), f1=r.get("f1"), total_cost=r["total_cost"],
                      cost_per_customer=r["cost_per_customer"]) for r in base_rows]
    for key, c in cands.items():
        m = ev.metrics_at(y_val, c["p_val"], c["thresholds"]["cost"])
        rk = ev.rank_metrics(y_val, c["p_val"], c["w"])
        c["val_cost"] = m["total_cost"]
        rank_rows.append(dict(model=key, pr_auc=rk["pr_auc"], roc_auc=rk["roc_auc"], recall=m["recall"],
                              precision=m["precision"], f1=m["f1"], total_cost=m["total_cost"],
                              cost_per_customer=m["cost_per_customer"]))
    pd.DataFrame(rank_rows).to_csv(tables / "validation_ranking.csv", index=False)
    best_key = min(cands, key=lambda k: cands[k]["val_cost"])
    best = cands[best_key]
    print(f"[dev] MÔ HÌNH ĐƯỢC CHỌN: {best_key}  thresholds={best['thresholds']}")

    # ---- hình trên validation
    ev.plot_roc_pr(y_val, {best_key: best["p_val"], "B3 – LR đơn giản": base_val_p["B3 – LR đơn giản (PAY_1, LIMIT_BAL)"]},
                   figs / "roc_pr_val.png")
    ev.plot_cost_curve({best_key: best["sw"],
                        "B3 – LR đơn giản": ev.sweep(y_val, base_val_p["B3 – LR đơn giản (PAY_1, LIMIT_BAL)"])},
                       nv, figs / "cost_curve_val.png")
    ev.plot_cm(ev.metrics_at(y_val, best["p_val"], best["thresholds"]["cost"]),
               f"Validation – {best_key}", figs / "cm_val.png")

    # ---- độ nhạy theo r, ma trận hối tiếc
    sens = ev.sensitivity(y_val, best["p_val"], best["w"], R_GRID)
    sens.to_csv(tables / "sensitivity_val.csv", index=False)
    ev.regret_matrix(y_val, best["p_val"], best["w"], R_GRID).to_csv(tables / "regret_val.csv")
    ev.plot_sensitivity(sens, figs / "sensitivity_val.png")

    # ---- permutation importance trên validation (thang PR-AUC, 5 lần lặp)
    pi = permutation_importance(best["est"], X_val, y_val, scoring="average_precision", n_repeats=5,
                                random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": X_val.columns, "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)
    imp.to_csv(tables / "permutation_importance_val.csv", index=False)
    ev.plot_importance(imp, figs / "permutation_importance_val.png")

    # ---- lưu mô hình được chọn và nhật ký
    joblib.dump(best["est"], models_dir / "selected.joblib")
    save_json(dict(model=best_key, algo=best["name"], mode=best["mode"], w=best["w"],
                   thresholds=best["thresholds"], best_params=best["best_params"]), out / "selected.json")
    update_run_log(out / "run_log.json", "data", info)
    update_run_log(out / "run_log.json", "split", dict(
        train=len(idx["train"]), val=len(idx["val"]), test=len(idx["test"]), sha256_test=idx["sha256_test"],
        positives=dict(train=int(y_tr.sum()), val=int(y_val.sum()))))
    update_run_log(out / "run_log.json", "environment", versions())
    update_run_log(out / "run_log.json", "dev", dict(selected=best_key, thresholds=best["thresholds"],
                                                    fast=bool(args.fast), seconds=round(time.time() - t_start, 1)))
    print(f"[dev] xong sau {time.time() - t_start:.0f}s. Bảng: {tables}  Hình: {figs}")


# ====================================================================== GIAI ĐOẠN FINAL
def stage_final(args):
    t_start = time.time()
    out, tables, figs, models_dir, split_path = _paths(args)
    sel = load_json(out / "selected.json")
    est = joblib.load(models_dir / "selected.joblib")
    df, _ = load_clean(args.data)
    df = add_features(df)
    idx = make_split(df, split_path)                       # kiểm tra mã băm SHA-256 của test
    X_tr, y_tr = get_xy(df, idx["train"]); X_val, y_val = get_xy(df, idx["val"]); X_te, y_te = get_xy(df, idx["test"])
    save_processed(df, idx, args.processed, parts=("test",))
    w = sel["w"]; thr = sel["thresholds"]
    p_tr, p_val, p_te = (est.predict_proba(X)[:, 1] for X in (X_tr, X_val, X_te))
    print(f"[final] mô hình {sel['model']} | test={len(X_te)} khách, {int(y_te.sum())} vỡ nợ")

    # ---- Bảng 3.7: đánh giá cuối trên test, bootstrap khoảng tin cậy
    n_boot = 100 if args.fast else 1000
    rows = []
    for c in CRITERIA:
        m = ev.metrics_at(y_te, p_te, thr[c])
        lo, hi = ev.bootstrap_cost_ci(y_te, p_te, thr[c], n_boot=n_boot)
        rows.append(dict(model=sel["model"], criterion=c, threshold=m["threshold"], recall=m["recall"],
                         precision=m["precision"], f1=m["f1"], total_cost=m["total_cost"],
                         cost_per_customer=m["cost_per_customer"], ci95_low=lo, ci95_high=hi,
                         saving=ev.saving(m["total_cost"], y_te)))
    final = pd.DataFrame(rows)
    final.to_csv(tables / "test_final.csv", index=False)
    nt = ev.naive_costs(y_te)
    save_json(nt, tables / "naive_costs_test.json")

    # ---- Bảng 3.9: đối chiếu tiêu chí thành công (ngưỡng tối ưu chi phí)
    c = final[final.criterion == "cost"].iloc[0]
    crit = pd.DataFrame([
        dict(metric="Recall", reference=">= 0.60", result=c.recall, passed=bool(c.recall >= 0.60)),
        dict(metric="Precision", reference=">= 0.35", result=c.precision, passed=bool(c.precision >= 0.35)),
        dict(metric="F1", reference=">= 0.44", result=c.f1, passed=bool(c.f1 >= 0.44))])
    crit.to_csv(tables / "success_criteria.csv", index=False)

    # ---- Bảng 3.11: train / validation / test
    tvt = []
    for name, y, p in (("Train", y_tr, p_tr), ("Validation", y_val, p_val), ("Test", y_te, p_te)):
        rk = ev.rank_metrics(y, p, w); m = ev.metrics_at(y, p, thr["cost"])
        tvt.append(dict(split=name, pr_auc=rk["pr_auc"], roc_auc=rk["roc_auc"], brier=rk["brier"],
                        cost_per_customer=m["cost_per_customer"]))
    pd.DataFrame(tvt).to_csv(tables / "train_val_test.csv", index=False)

    # ---- Bảng 3.10: lỗi theo nhóm khách hàng
    pred = (p_te >= thr["cost"]).astype(int)
    ev.error_by_group(y_te, pred, ev.make_groups(X_te, X_tr)).to_csv(tables / "error_by_group.csv", index=False)

    # ---- hình trên test
    ev.plot_cm(ev.metrics_at(y_te, p_te, thr["cost"]), f"Test – {sel['model']}", figs / "cm_test.png")
    ev.plot_residual_calibration(y_te, ev.unweight(p_te, w), figs / "residual_calibration_test.png")
    update_run_log(out / "run_log.json", "final", dict(model=sel["model"], thresholds=thr, test_sha256=idx["sha256_test"],
                                                       bootstrap=n_boot, seconds=round(time.time() - t_start, 1)))
    print(final.round(4).to_string(index=False))
    print(crit.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/raw/default of credit card clients.xls")
    ap.add_argument("--out", default="results")
    ap.add_argument("--stage", choices=["dev", "final"], required=True)
    ap.add_argument("--split", default="split_indices.json", help="chỉ số train/val/test (mặc định: split_indices.json ở thư mục gốc)")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--fast", action="store_true", help="kiểm tra nhanh mã nguồn, KHÔNG dùng để báo cáo")
    args = ap.parse_args()
    {"dev": stage_dev, "final": stage_final}[args.stage](args)


if __name__ == "__main__":
    main()
