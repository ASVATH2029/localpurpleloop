"""The supervised detector, plus an optional unsupervised second line.

Owner: ML Engineer #1 (plan Phase 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features import FEATURE_GROUPS, FEATURE_NAMES, feature_matrix


@dataclass
class Detector:
    """A trained classifier plus the metadata needed to reproduce its scores."""

    clf: object
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    iforest: object | None = None
    trained_rounds: int = 0
    n_train: int = 0
    model_kind: str = "random_forest"

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """P(attack) for each session row, in [0, 1]."""
        X = feature_matrix(df, self.feature_names)
        return self.clf.predict_proba(X)[:, 1]

    def anomaly_score(self, df: pd.DataFrame) -> np.ndarray:
        """Benign-only IsolationForest novelty score (higher = more unusual).

        Reported alongside the supervised score as a second line of defence for
        attacks that resemble nothing in the labelled corpus. It does not gate
        promotion -- it is diagnostic.
        """
        if self.iforest is None:
            return np.zeros(len(df))
        return -self.iforest.score_samples(feature_matrix(df, self.feature_names))

    def feature_importance(self, top: int = 15) -> pd.Series:
        if hasattr(self.clf, "feature_importances_"):
            vals = self.clf.feature_importances_
        elif hasattr(self.clf, "named_steps"):
            vals = np.abs(self.clf.named_steps["lr"].coef_[0])
        else:
            return pd.Series(dtype=float)
        return pd.Series(vals, index=self.feature_names).sort_values(ascending=False).head(top)


def _build_classifier(cfg: dict, seed: int):
    kind = cfg.get("model", "random_forest")
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(cfg.get("n_estimators", 300)),
            max_depth=cfg.get("max_depth") or None,
            min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=seed,
            n_jobs=-1,
        )
    if kind == "logistic_regression":
        return Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=2000,
                class_weight=cfg.get("class_weight", "balanced"),
                random_state=seed,
            )),
        ])
    raise ValueError(f"unknown detector model: {kind}")


def train(corpus: pd.DataFrame, cfg: dict, seed: int) -> Detector:
    """Fit a detector on the whole growing corpus."""
    if corpus.empty:
        raise ValueError("cannot train on an empty corpus")
    y = corpus["label"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("corpus has only one class; cannot train a detector")

    subset = cfg.get("feature_subset", "all")
    if subset not in FEATURE_GROUPS:
        raise ValueError(
            f"unknown feature_subset {subset!r}; expected one of {sorted(FEATURE_GROUPS)}"
        )
    names = FEATURE_GROUPS[subset]

    X = feature_matrix(corpus, names)
    clf = _build_classifier(cfg, seed)
    clf.fit(X, y)

    iforest = None
    if cfg.get("use_isolation_forest", False):
        benign = corpus.loc[corpus["label"] == 0]
        if len(benign) >= 50:
            iforest = IsolationForest(
                n_estimators=200, contamination="auto", random_state=seed, n_jobs=-1
            ).fit(feature_matrix(benign, names))

    return Detector(
        clf=clf,
        feature_names=names,
        iforest=iforest,
        trained_rounds=int(corpus["round"].max()),
        n_train=len(corpus),
        model_kind=cfg.get("model", "random_forest"),
    )
