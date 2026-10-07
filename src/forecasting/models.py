"""Point-forecast models: a schedule-profile baseline plus ML models, and the model wrapper.


* ProfileModel: the baseline (mean load per plant state x day type x quarter-hour).
* make_model(): creates Ridge, Random Forest, XGBoost (two settings) or LightGBM.
* fit_model(): trains a model; boosted models pick their tree count by early stopping on
  INNER training days (never on test data) and are then refit on all training rows.
* TrainedModel: wraps a fitted model with its feature list and the unseen-holiday rule, so
  every caller (validation, December forecast, service) predicts in exactly the same way.

Why the unseen-holiday rule exists: holidays in the history only occur in production
states, where they cut load by ~150 kW. Tree models add that drop on top of a SHUTDOWN and
forecast ~0 kW for 25-26 Dec, which is physically impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecasting.features import STATE_COLUMNS, day_type_of, plant_state_of, with_day_type

SEED = 42
PROFILE_COLUMNS = ["qh", "is_weekend", "is_holiday"] + STATE_COLUMNS
BOOSTED = {"xgboost_shallow", "xgboost", "lightgbm"}
MODEL_NAMES = ["profile", "ridge", "random_forest", "xgboost_shallow", "xgboost", "lightgbm"]


class ProfileModel(BaseEstimator, RegressorMixin):
    """Mean load per plant state x day type x quarter-hour, with fallbacks for unseen combinations."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProfileModel":
        df = pd.DataFrame({
            "state": plant_state_of(X).to_numpy(),
            "day": day_type_of(X).to_numpy(),
            "qh": X["qh"].to_numpy(),
            "y": np.asarray(y),
        })
        self.full_ = df.groupby(["state", "day", "qh"])["y"].mean()
        self.state_ = df.groupby(["state", "qh"])["y"].mean()
        self.day_ = df.groupby(["day", "qh"])["y"].mean()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        state = plant_state_of(X).to_numpy()
        day = day_type_of(X).to_numpy()
        qh = X["qh"].to_numpy()
        pred = self.full_.reindex(pd.MultiIndex.from_arrays([state, day, qh])).to_numpy()
        for table, keys in [(self.state_, [state, qh]), (self.day_, [day, qh])]:
            fallback = table.reindex(pd.MultiIndex.from_arrays(keys)).to_numpy()
            pred = np.where(np.isnan(pred), fallback, pred)
        return pred


def make_model(name: str, n_estimators: int | None = None, early_stopping: bool = False) -> Any:
    """Create an unfitted model. Boosted models use early stopping only when requested."""
    if name == "profile":
        return ProfileModel()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5,
                                     n_jobs=-1, random_state=SEED)
    if name in ("xgboost_shallow", "xgboost"):
        import xgboost as xgb

        if name == "xgboost_shallow":
            # The settings first suggested (depth 3, lr 0.01). 'reg:linear' is the old name of
            # 'reg:squarederror'; base_score=0.5 is dropped (recent versions estimate it).
            params = dict(n_estimators=1000, learning_rate=0.01, max_depth=3)
        else:  # deeper trees for the plant state x day type x time-of-day interactions
            params = dict(n_estimators=3000, learning_rate=0.05, max_depth=6, min_child_weight=5,
                          subsample=0.8, colsample_bytree=0.8)
        if n_estimators is not None:
            params["n_estimators"] = n_estimators
        if early_stopping:
            params["early_stopping_rounds"] = 50
        return xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                random_state=SEED, **params)
    if name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=n_estimators or 3000, learning_rate=0.05, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def early_stopping_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Inner validation rows for early stopping: every 10th calendar day of the training data.

    Using whole days (not random rows) avoids near-duplicate neighbours leaking
    between the fitting and early-stopping sets. The test period is never used.
    """
    day_number = (index.normalize() - index.normalize().min()).days.to_numpy()
    return day_number % 10 == 7


def holiday_weight(X: pd.DataFrame, y: pd.Series) -> float:
    """Where a holiday's load sits between a weekend (0) and a workday (1), estimated from training data.

    Averaged over the plant states that have holidays in the training data, weighted by holiday rows.
    """
    df = pd.DataFrame({"state": plant_state_of(X).to_numpy(), "day": day_type_of(X).to_numpy(),
                       "y": np.asarray(y)})
    means = df.groupby(["state", "day"])["y"].mean().unstack()
    weights, sizes = [], []
    for state, n in df[df["day"] == "holiday"].groupby("state").size().items():
        if {"workday", "weekend"} <= set(means.columns) and means.loc[state, ["workday", "weekend"]].notna().all():
            span = means.loc[state, "workday"] - means.loc[state, "weekend"]
            if span > 0:
                weights.append((means.loc[state, "holiday"] - means.loc[state, "weekend"]) / span)
                sizes.append(n)
    return float(np.clip(np.average(weights, weights=sizes), 0, 1)) if weights else 0.5


@dataclass
class TrainedModel:
    """A fitted model plus the feature list it expects and its rule for unseen holiday combinations.

    Rule: a holiday in a plant state that never had a holiday in training (e.g. SHUTDOWN on
    25-26 December) is forecast as weekend + w x (workday - weekend) for that state, with w
    estimated from the states where holidays were observed. Without this, tree models add the
    holiday effect learned in production states on top of a shutdown and predict ~0 kW.
    """

    name: str
    features: list[str]
    estimator: Any
    info: dict = field(default_factory=dict)
    seen_combos: set = field(default_factory=set)
    holiday_weight: float = 0.5

    def columns(self) -> list[str]:
        return PROFILE_COLUMNS if self.name == "profile" else self.features

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(X[self.columns()]), dtype=float)

    def unseen_holiday_mask(self, X: pd.DataFrame) -> np.ndarray:
        """Rows that are holidays in a plant state whose holiday behaviour was never observed."""
        state, day = plant_state_of(X).to_numpy(), day_type_of(X).to_numpy()
        return np.array([d == "holiday" and (s, d) not in self.seen_combos
                         and (s, "workday") in self.seen_combos and (s, "weekend") in self.seen_combos
                         for s, d in zip(state, day)], dtype=bool)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.raw_predict(X)
        unseen = self.unseen_holiday_mask(X)
        if unseen.any():
            weekend = self.raw_predict(with_day_type(X[unseen], "weekend"))
            workday = self.raw_predict(with_day_type(X[unseen], "workday"))
            pred[unseen] = weekend + self.holiday_weight * (workday - weekend)
        return np.clip(pred, 0, None)


def fit_model(name: str, X: pd.DataFrame, y: pd.Series, features: list[str]) -> TrainedModel:
    """Fit one model. Boosted models: find the tree count by early stopping, then refit on all rows."""
    cols = PROFILE_COLUMNS if name == "profile" else features
    combos = set(zip(plant_state_of(X).to_numpy(), day_type_of(X).to_numpy()))
    w = holiday_weight(X, y)
    if name not in BOOSTED:
        return TrainedModel(name, features, make_model(name).fit(X[cols], y),
                            seen_combos=combos, holiday_weight=w)

    es = early_stopping_mask(X.index)
    X_fit, y_fit, X_es, y_es = X.loc[~es, cols], y[~es], X.loc[es, cols], y[es]
    if name == "lightgbm":
        import lightgbm as lgb

        probe = make_model(name).fit(X_fit, y_fit, eval_set=[(X_es, y_es)],
                                     callbacks=[lgb.early_stopping(50, verbose=False)])
        best = int(probe.best_iteration_)
    else:
        probe = make_model(name, early_stopping=True).fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        best = int(probe.best_iteration) + 1
    final = make_model(name, n_estimators=max(best, 10)).fit(X[cols], y)
    return TrainedModel(name, features, final, {"n_estimators": best}, seen_combos=combos, holiday_weight=w)
