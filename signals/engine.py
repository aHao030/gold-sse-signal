"""每日开盘信号引擎：取数 → 构造隔夜金特征 → walk-forward 模型预测 → 落日志。

用法（北京时间）：
- 09:25 前后运行 scripts/run_signal.py：先同步黄金实时价与上证开盘快照
- 09:31 后信号即为当日有效（开盘进场口径）
dry-run 模式用本地缓存的最后一根分钟 bar 代替实时价，供盘后验证流程。
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from config import ENTRY_TIME, EXIT_TIME, LOGS_DIR, MIN_TRAIN_DAYS, ROLLING_WINDOW_DAYS, TZ
from data import gold, sse, store
from research.overnight import _price_at, build_minute_features
from signals.model import OvernightGoldModel

SIGNALS_CSV = LOGS_DIR / "signals.csv"


def _model_from_history(feat: pd.DataFrame) -> OvernightGoldModel:
    return OvernightGoldModel(min_train=MIN_TRAIN_DAYS,
                              window=ROLLING_WINDOW_DAYS).fit(
        feat["overnight_gold_ret"], feat["sse_intraday"])


def _history_features() -> pd.DataFrame:
    """用本地分钟数据构造历史（隔夜金收益, 上证日内收益）训练样本。"""
    sse_min = store.load("sse_1min")
    gold_min = store.load("xau_1min")
    if sse_min is None or gold_min is None:
        raise RuntimeError("本地缺少 1 分钟历史，请先运行 scripts/sync_daily.py 积累")
    days = sorted(sse_min["datetime"].dt.normalize().unique())
    return build_minute_features(sse_min, gold_min, days)


def run(dry_run: bool = False, debug: bool = False) -> dict:
    now = pd.Timestamp.now(tz=TZ)
    sse_daily = store.load("sse_daily")
    if sse_daily is None or len(sse_daily) == 0:
        raise RuntimeError("缺少上证日线（交易日历），先运行 scripts/sync_daily.py")
    today = now.normalize()
    hist_days = pd.DatetimeIndex(sorted(sse_daily["date"].dt.normalize()))
    prev_day = hist_days[hist_days < today]
    if len(prev_day) == 0:
        raise RuntimeError("日线中找不到今日之前的交易日")
    prev_day = prev_day[-1]

    # --- 隔夜金基准价：上一交易日 15:00 的金价 ---
    gold_min = store.load("xau_1min")
    t_prev_close = prev_day + pd.Timedelta(hours=15)
    g_base = _price_at(gold_min, t_prev_close) if gold_min is not None else None
    g_base_src = "xau_1min(15:00)"
    if g_base is None:  # 分钟数据不足时退回日线收盘
        xau_daily = store.load("xau_daily")
        prev_xau = xau_daily[xau_daily["date"].dt.normalize() <= prev_day] if xau_daily is not None else None
        g_base = float(prev_xau["close"].iloc[-1])
        g_base_src = "xau_daily(收盘近似)"

    # --- 当前金价 ---
    if dry_run:
        if gold_min is None or len(gold_min) == 0:
            raise RuntimeError("dry-run 需要 xau_1min 缓存")
        g_now = float(gold_min["close"].iloc[-1])
        g_now_src = f"dry-run(最后缓存 {gold_min['datetime'].iloc[-1]})"
    else:
        rt = gold.fetch_realtime(debug=debug)
        g_now = rt["price"]
        g_now_src = f"hf_XAU {rt['time']}"

    overnight_ret = g_now / g_base - 1

    # --- 上证开盘价 ---
    if dry_run:
        sse_min = store.load("sse_1min")
        today_bars = sse_min[sse_min["datetime"].dt.normalize() == pd.Timestamp(sse_min["datetime"].iloc[-1]).normalize()]
        sse_open = float(today_bars["open"].iloc[0])
        sse_src = "dry-run(缓存)"
    else:
        spot = sse.fetch_spot()
        sse_open = spot["open"]
        sse_src = f"spot {spot['time']}"

    # --- 模型训练与预测 ---
    feat = _history_features()
    feat = feat[feat["date"] < today]
    model = _model_from_history(feat)
    desc = model.describe()

    result = {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": today.strftime("%Y-%m-%d"),
        "gold_base": round(g_base, 4),
        "gold_base_src": g_base_src,
        "gold_now": round(g_now, 4),
        "gold_now_src": g_now_src,
        "overnight_gold_ret": round(overnight_ret, 6),
        "sse_open": sse_open,
        "sse_open_src": sse_src,
        "n_train": desc["n"],
        "model_ready": desc["ready"],
        "beta": round(desc["beta"], 6) if desc["ready"] else None,
        "t_beta": round(desc["t_beta"], 3) if desc["ready"] else None,
        "r2": round(desc["r2"], 5) if desc["ready"] else None,
        "pred_intraday_ret": None,
        "ci_low": None,
        "ci_high": None,
        "direction": None,
        "rule": f"开盘{ENTRY_TIME}进 → {EXIT_TIME}出，不留隔夜仓",
    }

    if model.ready:
        y_hat, lo, hi = model.predict_with_ci(overnight_ret)
        result.update({
            "pred_intraday_ret": round(y_hat, 6),
            "ci_low": round(lo, 6),
            "ci_high": round(hi, 6),
            "direction": "看涨" if y_hat > 0 else "看跌",
        })
    else:
        result["direction"] = "样本不足，无信号"

    # --- 落日志 ---
    row = pd.DataFrame([result])
    existing = SIGNALS_CSV.exists()
    row.to_csv(SIGNALS_CSV, mode="a", index=False, header=not existing,
               encoding="utf-8-sig")

    print(f"=== 黄金-上证 日内信号 {result['datetime']} ===")
    print(f"隔夜金收益: {overnight_ret:+.3%}  ({g_base_src} {g_base:.2f} → {g_now_src} {g_now:.2f})")
    print(f"上证开盘:   {sse_open:.2f} ({sse_src})")
    if model.ready:
        print(f"模型:       n={desc['n']} beta={desc['beta']:+.4f} t={desc['t_beta']:+.2f} R²={desc['r2']:.4f}")
        print(f"预测日内收益: {y_hat:+.3%}  [{lo:+.3%}, {hi:+.3%}]  → {result['direction']}")
    else:
        print(f"模型:       训练样本 {desc['n']} < {MIN_TRAIN_DAYS}，无信号")
    print(f"规则:       {result['rule']}")
    print(f"日志:       {SIGNALS_CSV}")
    return result


if __name__ == "__main__":
    run()
