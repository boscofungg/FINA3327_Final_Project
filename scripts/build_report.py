"""Compile the LaTeX report into the Reports directory."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_TEX = ROOT / "Reports" / "Breadth_Reversal_Report_Revised.tex"


def main():
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        REPORT_TEX.name,
    ]
    for _ in range(2):
        subprocess.run(command, cwd=REPORT_TEX.parent, check=True)


if __name__ == "__main__":
    main()
