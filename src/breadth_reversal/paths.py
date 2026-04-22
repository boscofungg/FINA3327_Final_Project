"""Project path helpers."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "Data"
GRAPHS_DIR = ROOT_DIR / "Graphs"
REPORTS_DIR = ROOT_DIR / "Reports"
