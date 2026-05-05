"""
train_props_model.py
────────────────────────────────────────────────────────────────────────────
Trains one XGBoost regression model per stat (PTS, REB, AST, FG3M, STL, BLK)
from player_features.csv.

Saves to src/NBA/models/:
  props_PTS.pkl  props_REB.pkl  props_AST.pkl
  props_FG3M.pkl props_STL.pkl  props_BLK.pkl

Each pkl is a dict:
  {
    "model":              XGBRegressor,
    "features":           [list of column names],
    "residual_std":       float,          ← global std (fallback)
    "player_std":         {name: float},  ← per-player std (preferred)
    "mae": float,
    "r2":  float,
  }

predict_props.py uses player_std[name] when available, else residual_std, to compute
P(stat > threshold) = 1 - norm.cdf(threshold, predicted, std).

Usage:
    python src/NBA/train_props_model.py
    python src/NBA/train_props_model.py --stats PTS REB
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

IN_PATH  = Path("src/NBA/data/player_features.csv")
OUT_DIR  = Path("src/NBA/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK"]

# Features shared across all stat models
BASE_FEATURES = [
    "IS_HOME",
    "IS_PLAYOFF",
    "GAME_IN_SERIES",
    "REST_DAYS",
    "SEASON_NUM",
    "OPP_DEF_RATING_L10",
]

# Per-stat rolling features (auto-expanded)
ROLLING_TEMPLATES = [
    "{stat}_L5", "{stat}_L10", "{stat}_L20",
    "MIN_L5", "MIN_L10",
    "FGA_L10",   # pace proxy
    "TOV_L10",   # turnover tendency
]


def feature_cols_for(stat: str, df_cols: list) -> list:
    cols = list(BASE_FEATURES)
    for tmpl in ROLLING_TEMPLATES:
        c = tmpl.format(stat=stat)
        if c in df_cols:
            cols.append(c)
    # add same-stat rolling for other related stats for context
    related = {"PTS": ["REB_L10", "AST_L10"],
               "REB": ["PTS_L10", "AST_L10"],
               "AST": ["PTS_L10", "REB_L10"],
               "FG3M": ["PTS_L10", "FGA_L10"],
               "STL": ["MIN_L10"],
               "BLK": ["MIN_L10", "REB_L10"]}
    for c in related.get(stat, []):
        if c in df_cols and c not in cols:
            cols.append(c)
    return [c for c in cols if c in df_cols]


def train_stat(df: pd.DataFrame, stat: str) -> dict:
    print(f"\n── Training {stat} model ──")
    target = stat
    if target not in df.columns:
        print(f"  Column {target} not found — skipping")
        return {}

    feat_cols = feature_cols_for(stat, list(df.columns))
    print(f"  Features ({len(feat_cols)}): {feat_cols}")

    needed_cols = feat_cols + [target, "PLAYER_ID", "PLAYER_NAME", "IS_PLAYOFF"]
    seen = set(); available = [c for c in needed_cols if c in df.columns and not (c in seen or seen.add(c))]
    sub = df[available].dropna()
    n_po = float(sub["IS_PLAYOFF"].sum())
    print(f"  Training rows: {len(sub):,}  (playoff: {n_po:.0f} = {n_po/len(sub):.1%})")

    X      = sub[feat_cols].values
    y      = sub[target].values
    groups = sub["PLAYER_ID"].values

    # Upweight playoff rows 5× so the model learns playoff-specific patterns.
    # Playoff games are only ~6% of rows but are exactly what we're predicting.
    PLAYOFF_WEIGHT = 5
    sample_weights = np.where(sub["IS_PLAYOFF"].values == 1, PLAYOFF_WEIGHT, 1.0)

    # Group-aware train/test split (hold out entire players)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    w_train         = sample_weights[train_idx]

    po_train = sub["IS_PLAYOFF"].values[train_idx].sum()
    po_test  = sub["IS_PLAYOFF"].values[test_idx].sum()
    print(f"  Train playoff rows: {po_train:,} / {len(train_idx):,}  "
          f"| Test playoff rows: {po_test:,} / {len(test_idx):,}")

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              sample_weight=w_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    preds = model.predict(X_test)
    residuals = y_test - preds
    residual_std = float(np.std(residuals))
    mae = float(mean_absolute_error(y_test, preds))
    r2  = float(r2_score(y_test, preds))

    # Per-player residual std on hold-out set (min 8 games to be reliable)
    test_names  = sub.iloc[test_idx]["PLAYER_NAME"].values
    player_std: dict[str, float] = {}
    for name in np.unique(test_names):
        mask = test_names == name
        if mask.sum() >= 8:
            player_std[name] = float(np.std(residuals[mask]))
    print(f"  MAE={mae:.2f}  R²={r2:.3f}  residual_std={residual_std:.2f}  "
          f"player_std computed for {len(player_std)} players")

    # Quick sanity: what's our calibration on a threshold?
    if stat == "PTS":
        threshold = 20.0
        predicted_probs = 1 - norm.cdf(threshold, preds, residual_std)
        actual_hit = (y_test >= threshold).astype(float)
        # bin into deciles
        bins = pd.cut(predicted_probs, bins=10)
        cal = pd.DataFrame({"pred_prob": predicted_probs, "actual": actual_hit, "bin": bins})
        cal_summary = cal.groupby("bin", observed=True).agg(
            pred_mean=("pred_prob", "mean"),
            actual_mean=("actual", "mean"),
            n=("actual", "count"),
        )
        print(f"\n  Calibration for {stat} > {threshold}:")
        print(cal_summary.to_string())

    return {
        "model":        model,
        "features":     feat_cols,
        "residual_std": residual_std,
        "player_std":   player_std,
        "mae":          mae,
        "r2":           r2,
    }


def main(stats=None):
    stats = stats or STATS
    print(f"Loading {IN_PATH} ...")
    df = pd.read_csv(IN_PATH, parse_dates=["GAME_DATE"])
    print(f"  {len(df):,} rows")

    for stat in stats:
        result = train_stat(df, stat)
        if not result:
            continue
        out_path = OUT_DIR / f"props_{stat}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved → {out_path}")

    print("\nAll models trained.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", nargs="+", default=None,
                        help="e.g. --stats PTS REB AST")
    args = parser.parse_args()
    main(args.stats)
