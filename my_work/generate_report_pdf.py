"""
generate_report_pdf.py — builds the official 2-page exam report PDF.

Delegates to scripts/build_exam_deliverables.py (fpdf2 + embedded figures).
Requires exactly 2 pages per exam rules.

Usage (from project root):
    python my_work/generate_report_pdf.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "build_exam_deliverables.py"


def main() -> None:
    result = subprocess.run([sys.executable, str(BUILD)], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)
    print("2-page report: deep_lense_project_report.pdf")
    print("10-slide deck: deep_lense_presentation.pdf")


if __name__ == "__main__":
    main()
