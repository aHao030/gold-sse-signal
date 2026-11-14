"""每日数据同步 CLI：收盘后（建议 15:10 后）运行，把各源增量并入本地 parquet。

用法：python scripts/sync_daily.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.sync import sync_all

if __name__ == "__main__":
    sync_all()
