"""上证指数数据：直连东方财富 push2his 官方端点（kline/trends2/stock）。

不走 akshare：akshare 的 index_code_id_map_em 依赖 clist 分页端点，
实测直连易被断连（RemoteDisconnected），kline/trends2 稳定得多。
时间戳统一为北京时间。
"""
from __future__ import annotations

import pandas as pd
import requests

from config import SSE_INDEX, TZ

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_TRENDS_URL = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"
_SNAP_URL = "https://push2.eastmoney.com/api/qt/stock/get"

# secid 规则：沪市指数 = 1.000001
_SECID = f"1.{SSE_INDEX}"


def _get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    data = (r.json() or {}).get("data")
    if data is None:
        raise RuntimeError(f"东财返回 data=null: {url} {params}")
    return data


def fetch_daily(start: str = "19900101", end: str = "22220101") -> pd.DataFrame:
    """上证指数日线全量（klt=101），1990 年至今约 8800 根。date 为北京时间 naive 日期。"""
    data = _get(_KLINE_URL, {
        "secid": _SECID, "klt": 101, "fqt": 1,
        "beg": start, "end": end, "lmt": 100000,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
    })
    rows = [k.split(",") for k in data["klines"]]
    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low",
                                     "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "close", "high", "low", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 列序对齐项目约定
    return df[["date", "open", "high", "low", "close", "volume", "amount"]]


def fetch_min(period: str = "1") -> pd.DataFrame:
    """上证指数分钟线。
    - period="1"：trends2 分时（东财只保留最近约 5 个交易日）
    - period="5" 等：kline klt=5（滚动窗口，实测约 36 个交易日）"""
    if period == "1":
        data = _get(_TRENDS_URL, {
            "secid": _SECID, "ndays": 5, "iscr": 0,
            "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        })
        rows = [k.split(",") for k in data["trades"]]
        df = pd.DataFrame(rows, columns=["datetime", "open", "close", "high",
                                         "low", "volume", "amount", "avg"])
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(TZ)
        for c in ("open", "close", "high", "low", "volume", "amount"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["datetime", "open", "high", "low", "close", "volume", "amount"]]

    klt = int(period)
    data = _get(_KLINE_URL, {
        "secid": _SECID, "klt": klt, "fqt": 1,
        "beg": 0, "end": 20500101, "lmt": 100000,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
    })
    rows = [k.split(",") for k in data["klines"]]
    df = pd.DataFrame(rows, columns=["datetime", "open", "close", "high",
                                     "low", "volume", "amount"])
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(TZ)
    for c in ("open", "close", "high", "low", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["datetime", "open", "high", "low", "close", "volume", "amount"]]


def fetch_spot() -> dict:
    """上证指数实时快照（push2 stock/get，秒级；fltt=2 表示已按小数返回）。"""
    data = _get(_SNAP_URL, {
        "secid": _SECID, "fltt": 2, "invt": 2,
        "fields": "f43,f44,f45,f46,f57,f58,f60,f86",
    })
    return {
        "name": data.get("f58", ""),
        "price": float(data["f43"]),
        "open": float(data["f46"]),
        "high": float(data["f44"]),
        "low": float(data["f45"]),
        "prev_close": float(data["f60"]),
        "time": pd.Timestamp.now(tz=TZ).strftime("%H:%M:%S"),
    }
