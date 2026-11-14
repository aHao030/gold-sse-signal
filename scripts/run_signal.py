"""每日开盘信号 CLI。

用法：
- 盘中实时：python scripts/run_signal.py            （09:25~09:31 运行最佳）
- 盘后验证：python scripts/run_signal.py --dry-run  （用本地缓存价代替实时）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signals.engine import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="用本地缓存数据空跑，不取实时")
    ap.add_argument("--debug", action="store_true", help="打印实时报价原始串（核对字段用）")
    args = ap.parse_args()
    run(dry_run=args.dry_run, debug=args.debug)
