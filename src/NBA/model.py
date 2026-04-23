import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

df = pd.read_csv("src/NBA/data/features.csv")

# Features and target
feature_cols = [
    "IS_HOME",
    "IS_PLAYOFF",
    "REST_DAYS",
    "WIN_PCT_L10",
    "OFF_RATING_L10",
    "DEF_RATING_L10",
    "PACE_PROXY_L10",
    "PM_TREND_L10",
    "OPP_WIN_PCT_L10",
    "OPP_OFF_RATING_L10",
    "OPP_DEF_RATING_L10",
    "OPP_PACE_PROXY_L10",
    "OPP_PM_TREND_L10",
    "SERIES_GAME_NUM",
    "SERIES_WINS",
    "SERIES_LOSSES",
]

df = df.dropna(subset=feature_cols + ["WIN"])
df = df.sort_values("GAME_DATE").reset_index(drop=True)

X = df[feature_cols]
y = df["WIN"]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train model
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model accuracy: {acc:.2%}")

# Feature importance
importances = dict(zip(feature_cols, model.feature_importances_))
sorted_feats = sorted(importances.items(), key=lambda x: x[1])

print("Feature importances:")
for feat, imp in reversed(sorted_feats):
    print(f"  {feat}: {imp:.4f}")

# Save feature importance chart
labels = [f for f, _ in sorted_feats]
values = [v for _, v in sorted_feats]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(labels, values, color="#1d428a")
ax.set_xlabel("Importance Score")
ax.set_title(f"XGBoost Feature Importances  |  Model Accuracy: {acc:.1%}")
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
plt.tight_layout()
os.makedirs("src/NBA/models", exist_ok=True)
chart_path = "src/NBA/models/feature_importance.png"
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"Feature importance chart saved to {chart_path}")

# Save model
os.makedirs("src/NBA/models", exist_ok=True)
with open("src/NBA/models/nba_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to src/NBA/models/nba_model.pkl")
