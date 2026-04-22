"""Workbook loaders for local analysis runs."""

from pathlib import Path

import pandas as pd

from .paths import DATA_DIR


DEFAULT_WORKBOOK = "Bloomberg_data.xlsx"


def load_data(workbook_name=DEFAULT_WORKBOOK):
    """Load and merge breadth and ES close data from the local workbook."""
    workbook_path = Path(workbook_name)
    if not workbook_path.is_absolute():
        workbook_path = DATA_DIR / workbook_name

    breadth = pd.read_excel(workbook_path, sheet_name="Breadth")
    es = pd.read_excel(workbook_path, sheet_name="ES Option ")

    breadth.columns = breadth.columns.str.strip()
    es.columns = es.columns.str.strip()

    df = breadth.merge(es, on="Date", how="inner")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").set_index("Date")
