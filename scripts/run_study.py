"""日线关系研究 CLI：20 年日线验证「隔夜金 → 上证日内」关系。

用法：python scripts/run_study.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.daily_study import run

if __name__ == "__main__":
    run()
