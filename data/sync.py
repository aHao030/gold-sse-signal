"""每日增量同步：把各数据源最新数据合并进本地 parquet（幂等，可重复跑）。

用法：收盘后（建议 15:10 之后）运行 scripts/sync_daily.py。
注意：东财接口有反爬 IP 软封禁（RemoteDisconnected），这里对东财请求做了
间隔节流；若仍被封，等 5~40 分钟再跑，不要连续重试。
"""
from __future__ import annotations

import time

import pandas as pd

from config import TZ
from data import gold, sse, store


def sync_sse_daily() -> int:
    """东财优先；被反爬封禁时自动降级 baostock。"""
    try:
        df = sse.fetch_daily()
    except Exception as exc:  # noqa: BLE001
        print(f"[sync] 东财日线失败（{type(exc).__name__}），降级 baostock ...")
        df = sse.fetch_daily_baostock()
    return store.save("sse_daily", df, key="date")


def _sse_min_with_fallback(period: str) -> pd.DataFrame:
    """东财分钟优先；被反爬封禁时降级新浪。"""
    try:
        return sse.fetch_min(period=period)
    except Exception as exc:  # noqa: BLE001
        print(f"[sync] 东财 {period} 分钟失败（{type(exc).__name__}），降级新浪 ...")
        return sse.fetch_min_sina(period=period)


def sync_sse_minute() -> dict:
    time.sleep(2)  # 东财反爬节流
    out = {}
    out["sse_1min"] = store.save("sse_1min", _sse_min_with_fallback("1"))
    time.sleep(2)
    out["sse_5min"] = store.save("sse_5min", _sse_min_with_fallback("5"))
    out["sse_60min"] = store.save("sse_60min", _sse_min_with_fallback("60"))
    return out


def sync_gold_daily() -> int:
    return store.save("xau_daily", gold.fetch_daily(), key="date")


def sync_gold_minute() -> dict:
    """GC=F：5m 取近 60 天、1m 取近 7 天（Yahoo range 档位）。"""
    out = {}
    out["xau_5min"] = store.save("xau_5min", gold.fetch_min("5m"))
    out["xau_1min"] = store.save("xau_1min", gold.fetch_min("1m"))
    return out


def sync_all() -> dict:
    """全部同步；单一数据源失败不影响其余（错误计入 summary["errors"]）。"""
    summary: dict = {}
    errors: dict = {}

    steps = [
        ("sse_daily", sync_sse_daily, {}),
        ("xau_daily", sync_gold_daily, {}),
        ("sse_minute", sync_sse_minute, None),
        ("gold_minute", sync_gold_minute, None),
    ]
    for name, fn, _ in steps:
        try:
            r = fn()
            if isinstance(r, dict):
                summary.update(r)
            else:
                summary[name] = r
        except Exception as exc:  # noqa: BLE001
            errors[name] = f"{type(exc).__name__}: {str(exc)[:120]}"
        time.sleep(1)

    if errors:
        summary["errors"] = errors
    print("--- 同步汇总（新增行数） ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary
