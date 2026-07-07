import pandas as pd

from scripts.run_temporal_backtest import make_expanding_time_folds, summarize_backtest


def test_make_expanding_time_folds_are_chronological():
    df = pd.DataFrame(
        {
            "FlightDate": pd.date_range("2024-01-01", periods=120, freq="h"),
            "ArrDel15": [0, 1] * 60,
        }
    )

    folds = make_expanding_time_folds(df, n_splits=3, min_train_fraction=0.5)

    assert len(folds) == 3
    for train, test in folds:
        assert train["FlightDate"].max() < test["FlightDate"].min()
        assert len(train) > 0
        assert len(test) > 0


def test_summarize_backtest_returns_metric_stats():
    folds = [
        {"metrics": {"roc_auc": 0.6, "pr_auc": 0.2, "f1": 0.3, "precision_at_top_10pct": 0.25, "lift_at_top_10pct": 1.4}},
        {"metrics": {"roc_auc": 0.7, "pr_auc": 0.25, "f1": 0.35, "precision_at_top_10pct": 0.30, "lift_at_top_10pct": 1.6}},
    ]

    summary = summarize_backtest(folds)

    assert summary["folds"] == 2
    assert abs(summary["metrics"]["roc_auc"]["mean"] - 0.65) < 1e-12
    assert "lift_at_top_10pct" in summary["metrics"]
