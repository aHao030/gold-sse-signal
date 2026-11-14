"""伦敦金数据：日线（新浪 XAU，2006 至今）/ 分钟（yfinance GC=F 优先、东财 GC00Y 备用）/ 实时（新浪 hf_XAU）。

注意：akshare/新浪体系内伦敦金分钟历史拿不到（调研已验证），分钟历史只能靠
yfinance 的有限窗口 + 每日增量落库自建。
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import requests

from config import GOLD_DAILY_SYMBOL, GOLD_MINUTE_SYMBOL, GOLD_EM_SECID, TZ

_EM_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------- 日线


def fetch_daily() -> pd.DataFrame:
    """新浪伦敦金现日线（akshare futures_foreign_hist），2006-11 至今约 20 年。
    清洗节假日填充行（OHLC 完全相同视为无效）。"""
    import akshare as ak

    df = ak.futures_foreign_hist(symbol=GOLD_DAILY_SYMBOL)
    df = df.rename(columns={"date": "date", "open": "open", "high": "high",
                            "low": "low", "close": "close", "volume": "volume"})
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"])
    df = _to_num(df, ["open", "high", "low", "close", "volume"])
    filler = (np.isclose(df["open"], df["close"])
              & np.isclose(df["high"], df["low"])
              & np.isclose(df["open"], df["high"]))
    df = df[~filler].reset_index(drop=True)
    return df


# ---------------------------------------------------------------- 分钟线

_YAHOO_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
_CRUMB_CACHE: dict = {"session": None, "crumb": None}


def _yahoo_session(refresh: bool = False) -> tuple:
    """带 cookie 的会话 + crumb（chart API 偶发 403，带 crumb 实测稳定）。"""
    if _CRUMB_CACHE["crumb"] and not refresh:
        return _CRUMB_CACHE["session"], _CRUMB_CACHE["crumb"]
    s = requests.Session()
    s.headers.update({"User-Agent": _YAHOO_UA})
    try:
        s.get("https://fc.yahoo.com", timeout=15)  # 拿 cookie，404 无所谓
    except Exception:
        pass
    r = s.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=15)
    crumb = r.text.strip()
    if r.status_code != 200 or not crumb or len(crumb) > 40:
        raise RuntimeError(f"获取 Yahoo crumb 失败: {r.status_code} {crumb[:30]!r}")
    _CRUMB_CACHE.update(session=s, crumb=crumb)
    return s, crumb


def fetch_min_yahoo(interval: str = "5m", start=None, end=None) -> pd.DataFrame:
    """COMEX 黄金主力 GC=F 分钟线——Yahoo chart API + cookie/crumb（实测 5m 取满 60 天）。
    Yahoo 限制：1m range 只给 7d 档（有效窗口约 30 天内）；5m 60 天。
    走系统代理（直连 403）。时间戳转北京时间。start/end 仅保留接口兼容。"""
    import pandas as pd

    range_map = {"1m": "7d", "5m": "60d", "15m": "60d", "60m": "730d"}
    last_exc: Exception | None = None
    r = None
    for attempt in range(3):
        try:
            s, crumb = _yahoo_session(refresh=attempt > 0)
            r = s.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/GC%3DF",
                params={"range": range_map.get(interval, "60d"), "interval": interval,
                        "includePrePost": "true", "crumb": crumb},
                timeout=30,
            )
            r.raise_for_status()
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(4 * (attempt + 1))
    else:
        raise RuntimeError(f"Yahoo chart 重试 3 次仍失败: {last_exc}")
    result = (r.json().get("chart") or {}).get("result") or []
    if not result:
        raise RuntimeError("Yahoo chart 返回空 result")
    res = result[0]
    ts = res.get("timestamp") or []
    q = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    df = pd.DataFrame({
        "ts": ts,
        "open": q.get("open"), "high": q.get("high"),
        "low": q.get("low"), "close": q.get("close"),
        "volume": q.get("volume"),
    }).dropna(subset=["close"])
    if df.empty:
        raise RuntimeError("Yahoo chart 无有效数据")
    df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(TZ)
    out = df[["datetime", "open", "high", "low", "close", "volume"]].drop_duplicates("datetime")
    return _to_num(out.sort_values("datetime").reset_index(drop=True),
                   ["open", "high", "low", "close", "volume"])


def fetch_min_yf(interval: str = "5m", start=None, end=None) -> pd.DataFrame:
    """COMEX 黄金主力 GC=F 分钟线（yfinance）。Yahoo 限制：1m 最近 30 天且单次 ≤7 天、
    5m 最近 60 天。时间戳统一转北京时间。"""
    import yfinance as yf

    df = yf.Ticker(GOLD_MINUTE_SYMBOL).history(interval=interval, start=start, end=end)
    if df is None or df.empty:
        raise RuntimeError("yfinance 返回空（可能被限流/网络不可达）")
    idx = df.index
    if idx.tz is None:
        dt = pd.to_datetime(idx).tz_localize("America/New_York").tz_convert(TZ)
    else:
        dt = idx.tz_convert(TZ)
    out = pd.DataFrame({
        "datetime": dt,
        "open": df["open"].values, "high": df["high"].values,
        "low": df["low"].values, "close": df["close"].values,
        "volume": df["volume"].values,
    }).reset_index(drop=True)
    return _to_num(out, ["open", "high", "low", "close", "volume"])


def fetch_min_em(klt: int = 5, lmt: int = 1300) -> pd.DataFrame:
    """东财 COMEX 金主力连续（secid=101.GC00Y）分钟线，akshare 未封装、直连 push2his。
    klt=5/15/30/60 理论可用（调研标注『未完全验证』，首次运行时核对）。时间戳按北京时间。"""
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": GOLD_EM_SECID, "klt": klt, "fqt": 1,
        "beg": 0, "end": 20500101, "lmt": lmt,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56",
    }
    r = requests.get(url, params=params, headers=_EM_HEADERS, timeout=20)
    data = (r.json() or {}).get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        raise RuntimeError("东财 GC00Y 分钟线返回空")
    rows = [k.split(",") for k in klines]
    df = pd.DataFrame(rows, columns=["datetime", "open", "close", "high", "low", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(TZ)
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    return _to_num(df, ["open", "high", "low", "close", "volume"])


def fetch_min(interval: str = "5m", start=None, end=None) -> pd.DataFrame:
    """分钟线统一入口：Yahoo chart API 直连优先 → EM GC00Y 兜底。
    （yfinance 库限流风险高，不再作为首选）"""
    try:
        return fetch_min_yahoo(interval, start=start, end=end)
    except Exception as exc:  # noqa: BLE001 - 记录降级原因后尝试备用源
        print(f"[gold] yahoo chart {interval} 失败（{exc}），降级东财 GC00Y ...")
    klt_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60}
    return fetch_min_em(klt=klt_map.get(interval, 5))


# ---------------------------------------------------------------- 实时


def fetch_realtime(debug: bool = False) -> dict:
    """伦敦金现（XAU/USD）实时报价。新浪 hf_XAU，秒级，必须带 Referer。
    字段顺序未逐位核实：取第一个能解析且落在合理区间(500~20000)的数值作为价格，
    首次运行时可用 debug=True 核对原始串。"""
    r = requests.get(
        "https://hq.sinajs.cn/list=hf_XAU",
        headers={"Referer": "https://finance.sina.com.cn/", **_EM_HEADERS},
        timeout=10,
    )
    r.encoding = "gbk"
    body = r.text
    if 'hf_XAU="' not in body:
        raise RuntimeError(f"新浪实时金报价异常: {body[:120]!r}")
    raw = body.split('hf_XAU="')[1].split('"')[0]
    if debug:
        print(f"[gold] hf_XAU 原始串: {raw!r}")
    fields = raw.split(",")
    price = None
    for f in fields:
        try:
            v = float(f)
        except ValueError:
            continue
        if 500 <= v <= 20000:
            price = v
            break
    if price is None:
        raise RuntimeError(f"未能从实时报价中解析出金价: {raw!r}")
    tmatch = re.search(r"\d{1,2}:\d{2}:\d{2}", raw)
    return {"price": price, "time": tmatch.group(0) if tmatch else "", "raw": raw}
