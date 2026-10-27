"""Baselines and a small neural network that corrects a stale prediction.

'Current system' stand-in: hold the last sounding's ducting prediction (persistence) -- what a
unit does when it runs the model once and keeps using the result. We test whether a small
neural network, using that stale sounding plus season/time, predicts the later state better.
We also report climatology so a gain over persistence is not mistaken for real skill.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import MODEL_FEATURES
from .features import TARGETS

CLASSIFICATION = {"duct_present"}
PERSIST_COL = {t: "src_" + t for t in TARGETS}


class Predictors:
    """Fits climatology, a neural-network ensemble and a gradient-boosting reference."""

    def __init__(self, train: pd.DataFrame, seeds: int = 5, features=None):
        self.train = train
        self.seeds = seeds
        self.features = list(features) if features is not None else list(MODEL_FEATURES)
        MODEL_FEATURES_ = self.features
        self.scaler = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(train[MODEL_FEATURES_].values)
        Xs = self.scaler.transform(train[MODEL_FEATURES_].values)
        self.nn, self.gbm = {}, {}
        for t in TARGETS:
            y = train[t].values
            if t in CLASSIFICATION:
                self.nn[t] = [MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, early_stopping=True,
                                            n_iter_no_change=15, max_iter=400, random_state=s).fit(Xs, y)
                              for s in range(seeds)]
                self.gbm[t] = HistGradientBoostingClassifier(random_state=0).fit(train[MODEL_FEATURES_], y)
            else:
                yl = np.log1p(y)
                self.nn[t] = [MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, early_stopping=True,
                                           n_iter_no_change=15, max_iter=400, random_state=s).fit(Xs, yl)
                              for s in range(seeds)]
                self.gbm[t] = HistGradientBoostingRegressor(random_state=0).fit(train[MODEL_FEATURES_], yl)
        self.log_max = {t: float(np.log1p(train[t].max())) for t in TARGETS if t not in CLASSIFICATION}
        self.clim = {t: train.groupby(["month", "hour"])[t].mean() for t in TARGETS}
        self.clim_all = {t: float(train[t].mean()) for t in TARGETS}

    def predict(self, pairs: pd.DataFrame) -> dict:
        """Return {target: {method: array}} plus {'nn_std': array} for the NN ensemble spread."""
        Xs = self.scaler.transform(pairs[self.features].values)
        out = {}
        for t in TARGETS:
            res = {"persistence": pairs[PERSIST_COL[t]].values.astype(float)}
            keys = list(zip(pairs["month"], pairs["hour"]))
            res["climatology"] = np.array([self.clim[t].get(k, self.clim_all[t]) for k in keys], dtype=float)
            if t in CLASSIFICATION:
                member = np.array([m.predict_proba(Xs)[:, 1] for m in self.nn[t]])
                res["nn"] = member.mean(0)
                res["nn_std"] = member.std(0)
                res["gbm"] = self.gbm[t].predict_proba(pairs[self.features])[:, 1]
            else:
                member = np.array([np.expm1(np.clip(m.predict(Xs), 0, self.log_max[t])) for m in self.nn[t]])
                res["nn"] = member.mean(0)
                res["nn_std"] = member.std(0)
                res["gbm"] = np.expm1(np.clip(self.gbm[t].predict(pairs[self.features]), 0, self.log_max[t]))
            out[t] = res
        return out


def _loss(target: str, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-sample loss: squared error on probability (Brier) for the duct flag, else abs error."""
    return (pred - y) ** 2 if target in CLASSIFICATION else np.abs(pred - y)


def block_bootstrap_ci(diff: np.ndarray, groups: np.ndarray, n: int = 1000, seed: int = 0) -> list:
    """95% CI of mean(diff), resampling whole calendar months to respect autocorrelation."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    sums = np.bincount(codes, weights=diff)
    cnts = np.bincount(codes)
    k = len(uniq)
    means = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, k, k)
        means[i] = sums[pick].sum() / cnts[pick].sum()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def score(test: pd.DataFrame, preds: dict, methods=("persistence", "climatology", "nn", "gbm")) -> dict:
    """Metrics per target and method, skill scores and bootstrap CIs for NN vs the baselines."""
    groups = test.index.to_period("M").astype(str).values
    out = {}
    for t in TARGETS:
        y = test[t].values.astype(float)
        res = {"n_test": int(len(y))}
        for m in methods:
            p = preds[t][m]
            if t in CLASSIFICATION:
                hard = (p >= 0.5).astype(int)
                res[m] = {"brier": float(np.mean((p - y) ** 2)),
                          "accuracy": float(accuracy_score(y, hard)),
                          "f1": float(f1_score(y, hard, zero_division=0))}
            else:
                res[m] = {"mae": float(np.mean(np.abs(p - y))),
                          "rmse": float(np.sqrt(np.mean((p - y) ** 2)))}
        key = "brier" if t in CLASSIFICATION else "rmse"
        sq = (lambda m: res[m]["brier"]) if t in CLASSIFICATION else (lambda m: res[m]["rmse"] ** 2)
        for m in methods:
            if m == "persistence":
                continue
            res[m]["skill_vs_persistence"] = float(1 - (sq(m) / sq("persistence")))
            if m != "climatology":
                res[m]["skill_vs_climatology"] = float(1 - (sq(m) / sq("climatology")))
        res["metric_for_skill"] = "Brier score" if t in CLASSIFICATION else "MSE"
        for base in ("persistence", "climatology"):
            d = _loss(t, y, preds[t]["nn"]) - _loss(t, y, preds[t][base])
            res[f"nn_minus_{base}_loss"] = {
                "mean": float(d.mean()), "ci95": block_bootstrap_ci(d, groups),
                "loss": "squared error of probability (Brier)" if t in CLASSIFICATION else "absolute error",
            }
        out[t] = res
    return out


def confidence_report(test: pd.DataFrame, preds: dict, target: str = "trap_strength_m") -> dict:
    """Does NN ensemble disagreement flag the predictions that turn out worse?"""
    y = test[target].values.astype(float)
    err = np.abs(preds[target]["nn"] - y)
    spread = preds[target]["nn_std"]
    rho = spearmanr(spread, err)
    q1, q2 = np.quantile(spread, [1 / 3, 2 / 3])
    band = np.where(spread <= q1, "green", np.where(spread <= q2, "yellow", "red"))
    return {"target": target, "spearman_spread_vs_error": float(rho.statistic),
            "mae_by_band": {b: float(err[band == b].mean()) for b in ("green", "yellow", "red")},
            "note": "bands are terciles of ensemble spread; error should rise green -> red if useful"}


def inference_latency_ms(predictors: Predictors, pairs: pd.DataFrame, reps: int = 50) -> float:
    """Average time to produce all three predictions for one new sounding, in milliseconds."""
    row = pairs.iloc[[0]]
    predictors.predict(row)  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        predictors.predict(row)
    return (time.perf_counter() - t0) / reps * 1000.0
