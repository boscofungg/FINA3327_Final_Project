"""Run the core backtest and print a compact summary."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from breadth_reversal import compute_signal, load_data, run_backtest
from breadth_reversal.backtest import TRUE_SHARPE_KEY


PARAMS = {
    "diff_lag": 1,
    "ma_window": 5,
    "pct_window": 20,
    "pct_threshold": 10.0,
    "holding_period": 1,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        default="Bloomberg_data.xlsx",
        help="Workbook filename inside Data/ or an absolute path.",
    )
    args = parser.parse_args()

    df = load_data(args.workbook)
    df = compute_signal(df, **PARAMS)
    _, stats = run_backtest(df)

    fields = [
        "Total Trades",
        "Win Rate (%)",
        "Compounded Return (%)",
        "Annualized Return (%)",
        "CAGR (%)",
        TRUE_SHARPE_KEY,
        "Active-Day Sharpe",
        "Exposure-Adjusted Annualized Return (%)",
        "Annualized Volatility (%)",
        "Maximum Drawdown (%)",
        "Exposure (%)",
        "Avg RF used (ann. %)",
        "BH Sharpe (ann. return - avg RF) / ann. vol",
    ]

    for field in fields:
        print(f"{field}: {stats[field]}")


if __name__ == "__main__":
    main()
