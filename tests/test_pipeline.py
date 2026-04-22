import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from breadth_reversal.backtest import TRUE_SHARPE_KEY, run_backtest
from breadth_reversal.compute_breadth import compute_signal
from breadth_reversal.data_loader import load_data

DATA_FILE = ROOT / "Data" / "Bloomberg_data.xlsx"
PARAMS = {
    "diff_lag": 1,
    "ma_window": 5,
    "pct_window": 20,
    "pct_threshold": 10.0,
    "holding_period": 1,
}


def load_workbook_data():
    return load_data(DATA_FILE)


class PipelineRegressionTests(unittest.TestCase):
    def test_workbook_sheets_merge_to_expected_shape(self):
        df = load_workbook_data()
        self.assertEqual(df.shape, (4063, 2))
        self.assertEqual(list(df.columns), ["Breadth", "ES_Close"])

    def test_signal_and_backtest_core_outputs_remain_stable(self):
        df = compute_signal(load_workbook_data(), **PARAMS)
        _, stats = run_backtest(df)

        self.assertEqual(stats["Total Trades"], 273)
        self.assertTrue(math.isclose(stats["Exposure (%)"], 13.91, rel_tol=0, abs_tol=0.01))
        self.assertTrue(math.isclose(stats["Annualized Return (%)"], 5.35, rel_tol=0, abs_tol=0.01))
        self.assertTrue(math.isclose(stats["Annualized Volatility (%)"], 7.20, rel_tol=0, abs_tol=0.01))
        self.assertTrue(math.isclose(stats["Avg RF used (ann. %)"], 1.41, rel_tol=0, abs_tol=0.01))
        self.assertTrue(math.isclose(stats["BH Annualized Return (%)"], 13.22, rel_tol=0, abs_tol=0.01))
        self.assertTrue(math.isclose(stats["Capital Efficiency Ratio"], 9.07, rel_tol=0, abs_tol=0.01))
        self.assertTrue(math.isclose(stats[TRUE_SHARPE_KEY], 0.55, rel_tol=0, abs_tol=0.01))


if __name__ == "__main__":
    unittest.main()
