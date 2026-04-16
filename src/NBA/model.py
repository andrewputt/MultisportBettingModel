import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

df = pd.read_csv("src/NBA/data/features.csv")

# Features and target
feature_cols = [
    "IS_HOME",
    "REST_DAYS",
    "WIN_PCT_L10",
    "OFF_RATING_L10",
    "DEF_RATING_L10",
    "PACE_PROXY_L10",
    "PM_TREND_L10",
]

df = df.dropna(subset=feature_cols + ["WIN"])

X = df[feature_cols]
y = df["WIN"]

df = df.sort_values("GAME_DATE")
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model accuracy: {acc:.2%}")

# Save model
os.makedirs("src/NBA/models", exist_ok=True)
with open("src/NBA/models/nba_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to src/NBA/models/nba_model.pkl")
