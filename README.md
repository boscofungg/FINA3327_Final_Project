# Breadth Reversal Analysis

Project layout:

- `Data/`: source workbook and Fama-French files
- `Graphs/`: generated chart outputs
- `Reports/`: LaTeX report and compiled PDF
- `src/breadth_reversal/`: reusable analysis package
- `scripts/`: local entrypoints
- `tests/`: regression coverage

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the analysis

```bash
python3 scripts/run_analysis.py
python3 scripts/generate_charts.py
python3 -m unittest discover -s tests -v
#python3 scripts/build_report.py
```

Notes:

- `scripts/run_analysis.py` prints the headline metrics, including the requested true Sharpe formula and the exposure-adjusted annualized return.
- `scripts/generate_charts.py` writes all figures into `Graphs/`.
- `scripts/build_report.py` compiles `Reports/Breadth_Reversal_Report_Revised.tex` in place and keeps the relative figure links working from the `Reports/` directory.
