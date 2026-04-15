import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import os
from datetime import datetime

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train a LogisticRegression model and evaluate on test set.
    Returns model, predictions, and metrics.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, preds)
    
    # Handle edge cases where all predictions are one class
    try:
        precision = precision_score(y_test, preds, zero_division=0)
    except:
        precision = 0.0
    
    try:
        recall = recall_score(y_test, preds, zero_division=0)
    except:
        recall = 0.0
    
    return model, preds, accuracy, precision, recall


def simulate_roi(y_test, preds, bet_amount=100):
    """
    Simulate ROI assuming $bet_amount on each model prediction at even odds (1.0).
    Returns total P&L and ROI percentage.
    """
    wins = (preds == y_test).sum()
    total_bet = len(preds) * bet_amount
    total_return = wins * (bet_amount * 2)  # $100 bet returns $200 on win (even odds)
    pnl = total_return - total_bet
    roi = (pnl / total_bet * 100) if total_bet > 0 else 0.0
    return pnl, roi


def run_walk_forward_backtest():
    """
    Execute walk-forward backtest on NBA game data.
    Trains on all prior months, tests on current month, rolls forward.
    Logs results to CSV and prints summary.
    """
    # Load features
    df = pd.read_csv("src/NBA/data/features.csv")
    
    # Parse GAME_DATE and drop rows with NaN in critical columns
    feature_cols = [
        "IS_HOME",
        "REST_DAYS",
        "WIN_PCT_L10",
        "OFF_RATING_L10",
        "DEF_RATING_L10",
        "PACE_PROXY_L10",
        "PM_TREND_L10",
    ]
    
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.dropna(subset=feature_cols + ["WIN"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    
    if len(df) == 0:
        print("❌ No valid data found after dropping NaNs.")
        return
    
    # Extract year-month for grouping
    df["YEAR_MONTH"] = df["GAME_DATE"].dt.to_period("M")
    
    # Get unique months in chronological order
    months = sorted(df["YEAR_MONTH"].unique())
    
    if len(months) < 2:
        print(f"⚠️  Only {len(months)} month(s) available. Walk-forward backtest requires at least 2 months.")
        return
    
    print(f"\n📊 Walk-Forward Backtest: {len(months)} months from {months[0]} to {months[-1]}")
    print("=" * 80)
    
    results = []
    all_test_preds = []
    all_test_actuals = []
    final_model = None
    
    # Walk-forward loop
    for i, test_month in enumerate(months[1:], start=1):  # Start from month 2
        train_mask = df["YEAR_MONTH"] < test_month
        test_mask = df["YEAR_MONTH"] == test_month
        
        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, "WIN"]
        X_test = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, "WIN"]
        
        # Skip if no training or test data
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"⏭️  {test_month}: Skipping (insufficient data)")
            continue
        
        # Train and evaluate
        model, preds, accuracy, precision, recall = train_and_evaluate(
            X_train, y_train, X_test, y_test
        )
        pnl, roi = simulate_roi(y_test, preds)
        
        # Store results
        num_games = len(y_test)
        results.append({
            "MONTH": str(test_month),
            "NUM_GAMES": num_games,
            "ACCURACY": round(accuracy, 4),
            "PRECISION": round(precision, 4),
            "RECALL": round(recall, 4),
            "SIMULATED_PNL": round(pnl, 2),
            "SIMULATED_ROI": round(roi, 2),
        })
        
        all_test_preds.extend(preds)
        all_test_actuals.extend(y_test)
        final_model = model
        
        print(f"✅ {test_month}: Accuracy={accuracy:.2%} | Precision={precision:.2%} | Recall={recall:.2%} | ROI={roi:.2f}% | P&L=${pnl:,.2f} ({num_games} games)")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs("src/NBA/data", exist_ok=True)
    results_df.to_csv("src/NBA/data/backtest_results.csv", index=False)
    print("\n✅ Backtest results saved to src/NBA/data/backtest_results.csv")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("📈 Summary Statistics:")
    print(f"  Total Months Tested: {len(results)}")
    print(f"  Avg Accuracy: {results_df['ACCURACY'].mean():.2%}")
    print(f"  Avg Precision: {results_df['PRECISION'].mean():.2%}")
    print(f"  Avg Recall: {results_df['RECALL'].mean():.2%}")
    print(f"  Total Simulated P&L: ${results_df['SIMULATED_PNL'].sum():,.2f}")
    print(f"  Avg Monthly ROI: {results_df['SIMULATED_ROI'].mean():.2f}%")
    print(f"  Best Month: {results_df.loc[results_df['ACCURACY'].idxmax(), 'MONTH']} ({results_df['ACCURACY'].max():.2%})")
    print(f"  Worst Month: {results_df.loc[results_df['ACCURACY'].idxmin(), 'MONTH']} ({results_df['ACCURACY'].min():.2%})")
    
    # Save final model trained on all data
    if final_model is not None:
        # Re-train on all available data
        X_all = df[feature_cols]
        y_all = df["WIN"]
        final_model_all = LogisticRegression(max_iter=1000, random_state=42)
        final_model_all.fit(X_all, y_all)
        
        os.makedirs("src/NBA/models", exist_ok=True)
        with open("src/NBA/models/nba_model.pkl", "wb") as f:
            pickle.dump(final_model_all, f)
        print(f"\n✅ Final model trained on all data and saved to src/NBA/models/nba_model.pkl")
    
    # Save backtest metadata
    metadata = {
        "backtest_date": datetime.now().isoformat(),
        "months_tested": len(results),
        "total_games": len(df),
        "features_used": feature_cols,
        "avg_accuracy": float(results_df['ACCURACY'].mean()),
        "total_simulated_pnl": float(results_df['SIMULATED_PNL'].sum()),
    }
    
    import json
    with open("src/NBA/data/backtest_report.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("✅ Backtest metadata saved to src/NBA/data/backtest_report.json")


if __name__ == "__main__":
    run_walk_forward_backtest()
