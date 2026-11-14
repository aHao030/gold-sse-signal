"""分钟级 walk-forward 回测：开盘 09:31 进场、14:55 出场，不留隔夜仓。

严格 walk-forward：预测第 t 天只用 t 之前的数据拟合（滚动/扩展 OLS）。
数据不足 MIN_TRAIN_DAYS 时该日不给信号（记为空仓）。
初始 1 分钟样本仅最近几个交易日（东财限制），样本量限制会明确写进报告。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MIN_TRAIN_DAYS, MPL_FONT, REPORTS_DIR, ROLLING_WINDOW_DAYS
from data import store
from signals.model import OvernightGoldModel
from research.overnight import build_minute_features

plt.rcParams["font.sans-serif"] = [MPL_FONT]
plt.rcParams["axes.unicode_minus"] = False


def run(freq: str = "1m") -> str:
    sym_sse = "sse_1min" if freq == "1m" else "sse_5min"
    sym_gold = "xau_1min" if freq == "1m" else "xau_5min"
    sse_min = store.load(sym_sse)
    gold_min = store.load(sym_gold)
    sse_daily = store.load("sse_daily")
    if sse_min is None or gold_min is None:
        raise RuntimeError(f"本地缺少分钟数据（{sym_sse}/{sym_gold}），请先运行 scripts/sync_daily.py")

    # 上证交易日历（以本地日线为准）
    days = sorted(sse_min["datetime"].dt.normalize().unique())
    feat = build_minute_features(sse_min, gold_min, days)
    n_days = len(feat)
    if n_days == 0:
        raise RuntimeError("有效交易日为 0：分钟数据与交易日历对不上")

    # --- walk-forward ---
    records = []
    feat_sorted = feat.sort_values("date").reset_index(drop=True)
    for i in range(n_days):
        train = feat_sorted.iloc[:i]
        model = OvernightGoldModel(min_train=MIN_TRAIN_DAYS,
                                   window=ROLLING_WINDOW_DAYS).fit(
            train["overnight_gold_ret"], train["sse_intraday"])
        row = feat_sorted.iloc[i]
        rec = {"date": row["date"], "x": row["overnight_gold_ret"],
               "realized": row["sse_intraday"], "entry": row["entry"],
               "exit": row["exit"], "n_train": model.n_}
        if model.ready:
            rec["y_hat"] = model.predict(row["overnight_gold_ret"])
            rec["beta"] = model.describe()["beta"]
        else:
            rec["y_hat"] = np.nan
        records.append(rec)
    res = pd.DataFrame(records)
    valid = res.dropna(subset=["y_hat"]).copy()
    n_signal = len(valid)
    if n_signal == 0:
        raise RuntimeError(f"无信号日：样本 {n_days} 天 < MIN_TRAIN_DAYS={MIN_TRAIN_DAYS}，继续每日同步积累数据")

    valid["hit"] = np.sign(valid["y_hat"]) == np.sign(valid["realized"])
    valid["pnl_follow"] = np.where(valid["y_hat"] > 0, valid["realized"], 0.0)  # 只做多：信号偏多才持有
    valid["pnl_hold"] = valid["realized"]                                       # 基准：每天开盘持有到尾盘
    valid["ic_prod"] = valid["y_hat"] * valid["realized"]

    ic = np.corrcoef(valid["y_hat"], valid["realized"])[0, 1]
    metrics = {
        "总交易日": n_days,
        "有信号日": n_signal,
        "方向命中率": valid["hit"].mean(),
        "IC(预测vs实现)": ic,
        "跟信号日均收益": valid["pnl_follow"].mean(),
        "每日持有基准日均收益": valid["pnl_hold"].mean(),
        "跟信号累计收益": (1 + valid["pnl_follow"]).prod() - 1,
        "基准累计收益": (1 + valid["pnl_hold"]).prod() - 1,
        "beta均值": valid["beta"].mean(),
    }

    # --- 图：净值曲线 + 散点 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = valid["date"]
    axes[0].plot(x, (1 + valid["pnl_follow"]).cumprod(), label="跟信号(开盘进,尾盘出)")
    axes[0].plot(x, (1 + valid["pnl_hold"]).cumprod(), label="基准(每日持有)", alpha=0.7)
    axes[0].legend(); axes[0].set_title("累计净值"); axes[0].tick_params(axis="x", rotation=30)
    axes[1].scatter(valid["x"] * 100, valid["realized"] * 100, s=14, alpha=0.6)
    b = np.polyfit(valid["x"], valid["realized"], 1)
    xs = np.linspace(valid["x"].min(), valid["x"].max(), 50)
    axes[1].plot(xs * 100, np.polyval(b, xs) * 100, "r--", lw=1)
    axes[1].set_xlabel("隔夜金收益 %"); axes[1].set_ylabel("上证日内收益 %")
    axes[1].set_title(f"实现收益 vs 信号（IC={ic:.3f}）")
    fig.tight_layout()
    p1 = REPORTS_DIR / f"minute_backtest_{freq}.png"
    fig.savefig(p1, dpi=120)

    # --- 报告 ---
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    md = [
        f"# 分钟级回测报告（{today}，频率={freq}）",
        "",
        f"- 数据窗口：{res['date'].min():%Y-%m-%d} ~ {res['date'].max():%Y-%m-%d}",
        f"- 进出场规则：09:31 首根 bar 收盘价进场 → 14:55/收盘前最后一根 bar 收盘价出场，不留隔夜仓",
        f"- 模型：OLS（上证日内收益 ~ 隔夜金收益），min_train={MIN_TRAIN_DAYS}，"
        f"rolling={ROLLING_WINDOW_DAYS}，严格 walk-forward",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items() if isinstance(v, float)),
        "\n".join(f"| {k} | {int(v)} |" for k, v in metrics.items() if not isinstance(v, float)),
        "",
        "## 样本量警告",
        "",
        f"- 当前 1 分钟历史仅东财最近约 5 个交易日（调研验证），5 分钟约 36 个交易日；"
        f"本报告有信号日仅 **{n_signal}** 天，统计意义有限，结论仅供观察趋势，"
        "需持续 `sync_daily.py` 积累数据后复核",
        "",
        f"![回测](minute_backtest_{freq}.png)",
        "",
        "- 本报告不构成投资建议",
    ]
    out = REPORTS_DIR / f"minute_backtest_{freq}_{pd.Timestamp.now():%Y%m%d}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    res.to_csv(REPORTS_DIR / f"minute_backtest_{freq}_daily.csv", index=False)
    print(f"[backtest:{freq}] 报告: {out}")
    print(f"[backtest:{freq}] " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
    return str(out)
