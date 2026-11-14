"""特征构造：隔夜金收益 → 上证日内收益。

两种口径：
- 日线代理：用于 20 年长历史关系研究（近似，见 daily_study 说明）
- 分钟精确：15:00（上一交易日）→ 次日 09:31 的伦敦金收益，用于日内系统
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ENTRY_TIME, EXIT_TIME, TZ


def build_daily_features(sse_daily: pd.DataFrame,
                         xau_daily: pd.DataFrame) -> pd.DataFrame:
    """日线代理特征。预测目标：sse_intraday = close/open - 1（开盘进、收盘出）。
    特征：
    - gold_ret_prev：上一伦敦金交易日 close-to-close 收益（滞后一日代理，开盘时已知）
    - gold_ret_same：同日金的收益（含盘后信息，仅作对照，不用于信号）
    - gold_vol20_prev：金 20 日波动率（regime 分组用）
    - sse_gap：开盘缺口（昨收→今开）"""
    sse = sse_daily.sort_values("date").copy()
    sse["prev_close"] = sse["close"].shift(1)
    sse["sse_gap"] = sse["open"] / sse["prev_close"] - 1
    sse["sse_intraday"] = sse["close"] / sse["open"] - 1

    xau = xau_daily.sort_values("date").copy()
    xau["gold_ret"] = xau["close"] / xau["close"].shift(1) - 1
    xau["gold_ret_prev"] = xau["gold_ret"].shift(1)
    xau["gold_vol20"] = xau["gold_ret"].rolling(20).std()
    xau["gold_vol20_prev"] = xau["gold_vol20"].shift(1)

    # 按日期直接左连：A股休市日金的行不会出现在上证侧（NaN 由模型/统计自动丢弃）
    df = sse[["date", "sse_gap", "sse_intraday"]].merge(
        xau[["date", "gold_ret_prev", "gold_ret", "gold_vol20_prev"]],
        on="date", how="left",
    )
    df = df.rename(columns={"gold_ret": "gold_ret_same"})
    df["year"] = df["date"].dt.year
    return df


def _price_at(gold_min: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    """gold_min 按 datetime 升序；返回时间戳 <= ts 的最后一根 bar 的 close。"""
    if gold_min is None or len(gold_min) == 0:
        return None
    idx = gold_min["datetime"].searchsorted(ts, side="right") - 1
    if idx < 0:
        return None
    return float(gold_min["close"].iloc[idx])


def build_minute_features(sse_min: pd.DataFrame, gold_min: pd.DataFrame,
                          trading_days: list) -> pd.DataFrame:
    """分钟精确特征：每个上证交易日一行。
    - overnight_gold_ret：上一交易日 15:00 → 今日 09:31 伦敦金收益（精确隔夜窗口）
    - entry（09:31 bar close）/ exit（<=14:55 的最后一根 bar close）
    - sse_intraday = exit/entry - 1
    - sse_gap = 今日首根 bar open / 昨日收盘 - 1"""
    days = [pd.Timestamp(d).tz_localize(TZ) for d in trading_days]
    rows = []
    sse_by_day = {d: g for d, g in sse_min.groupby(sse_min["datetime"].dt.normalize())}
    for i, day in enumerate(days):
        if i == 0:
            continue
        prev = days[i - 1]
        bars = sse_by_day.get(day)
        if bars is None or len(bars) < 10:
            continue
        t_entry = day + pd.Timedelta(hours=int(ENTRY_TIME[:2]), minutes=int(ENTRY_TIME[3:]))
        t_exit = day + pd.Timedelta(hours=int(EXIT_TIME[:2]), minutes=int(EXIT_TIME[3:]))
        t_prev_close = prev + pd.Timedelta(hours=15)

        g_base = _price_at(gold_min, t_prev_close)
        g_now = _price_at(gold_min, t_entry)
        if g_base is None or g_now is None:
            continue

        entry_bars = bars[bars["datetime"] <= t_entry + pd.Timedelta(minutes=1)]
        if entry_bars.empty:
            continue
        entry = float(entry_bars["close"].iloc[0])
        entry_open = float(entry_bars["open"].iloc[0])

        exit_bars = bars[bars["datetime"] <= t_exit]
        exit_px = float(exit_bars["close"].iloc[-1]) if not exit_bars.empty else float(bars["close"].iloc[-1])

        prev_bars = sse_by_day.get(prev)
        prev_close = float(prev_bars["close"].iloc[-1]) if prev_bars is not None else np.nan

        rows.append({
            "date": day,
            "overnight_gold_ret": g_now / g_base - 1,
            "entry": entry,
            "exit": exit_px,
            "sse_intraday": exit_px / entry - 1,
            "sse_gap": entry_open / prev_close - 1 if np.isfinite(prev_close) else np.nan,
        })
    return pd.DataFrame(rows)
