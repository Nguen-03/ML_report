"""Tiền xử lý, Pipeline, không gian tìm kiếm siêu tham số và các mô hình cơ sở."""
import numpy as np
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .utils import CAT, C_FN, C_FP, MONEY, SEED


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def make_preprocessor(kind, feature_cols):
    num_other = [c for c in feature_cols if c not in CAT + MONEY]
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    money_cols = [c for c in MONEY if c in feature_cols]
    if kind == "linear":   # LR: log đối xứng + z-score cho biến tiền tệ
        money = Pipeline([("log", FunctionTransformer(signed_log)), ("sc", StandardScaler())])
        return ColumnTransformer([("cat", cat, CAT),
                                  ("money", money, money_cols),
                                  ("num", StandardScaler(), num_other)])
    # Mô hình cây: one-hot cho biến phân loại, giữ nguyên các biến còn lại
    return ColumnTransformer([("cat", cat, CAT)], remainder="passthrough")


def class_weight_for(mode, n0, n1):
    return {"none": None,
            "balanced": {0: 1.0, 1: n0 / n1},
            "cost": {0: 1.0, 1: C_FN / C_FP}}[mode]


def weight_value(mode, n0, n1):
    """Hệ số w của lớp vỡ nợ ứng với từng chế độ trọng số."""
    return {"none": 1.0, "balanced": n0 / n1, "cost": C_FN / C_FP}[mode]


def build_pipeline(name, mode, feature_cols, n0, n1):
    cw = class_weight_for(mode, n0, n1)
    if name == "LR":
        clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000,
                                 class_weight=cw, random_state=SEED)
        kind = "linear"
    elif name == "RF":
        if mode == "balanced":
            cw = "balanced_subsample"
        clf = RandomForestClassifier(class_weight=cw, n_jobs=-1, random_state=SEED)
        kind = "tree"
    else:
        clf = HistGradientBoostingClassifier(class_weight=cw, random_state=SEED)
        kind = "tree"
    prep = make_preprocessor(kind, feature_cols)   # fit trong pipeline => không rò rỉ
    return Pipeline([("prep", prep), ("clf", clf)])


def search_space(name):
    """Không gian tìm kiếm (Bảng 2.10)."""
    if name == "LR":
        return {"clf__C": loguniform(1e-3, 1e2)}
    if name == "RF":
        return {"clf__n_estimators": [200, 300, 500],
                "clf__max_depth": [6, 8, 12, None],
                "clf__min_samples_leaf": [5, 10, 20, 40],
                "clf__max_features": ["sqrt", 0.3, 0.5]}
    return {"clf__learning_rate": loguniform(0.02, 0.2),
            "clf__max_iter": [150, 250, 400],
            "clf__max_leaf_nodes": [8, 15, 31],
            "clf__min_samples_leaf": [20, 50, 100],
            "clf__l2_regularization": loguniform(1e-3, 10)}


N_ITER = {"LR": 8, "RF": 20, "GB": 20}


class PayRuleBaseline(BaseEstimator, ClassifierMixin):
    """B2: cảnh báo mọi khách đang trễ hạn ở kỳ gần nhất (PAY_1 >= 1); không học tham số."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        flag = (np.asarray(X["PAY_1"]) >= 1).astype(float)
        return np.column_stack([1 - flag, flag])

    def predict(self, X):
        return (np.asarray(X["PAY_1"]) >= 1).astype(int)


def baselines():
    """B1: lớp đa số; B2: luật PAY_1 >= 1; B3: LR đơn giản (PAY_1, LIMIT_BAL)."""
    b3 = Pipeline([
        ("prep", ColumnTransformer([("s", StandardScaler(), ["PAY_1", "LIMIT_BAL"])], remainder="drop")),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    return {"B1 – Lớp đa số": DummyClassifier(strategy="most_frequent"),
            "B2 – Luật PAY_1 ≥ 1": PayRuleBaseline(),
            "B3 – LR đơn giản (PAY_1, LIMIT_BAL)": b3}
