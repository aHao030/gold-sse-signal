"""分钟级回测 CLI（walk-forward，开盘进/尾盘出）。

用法：python scripts/run_backtest.py [--freq 1m|5m]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.minute_backtest import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", choices=["1m", "5m"], default="1m",
                    help="回测频率（1m 样本当前仅约5个交易日）")
    args = ap.parse_args()
    run(freq=args.freq)
