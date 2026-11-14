"""20 年日线研究：隔夜金收益 vs 上证日内收益——验证关系是否存在、是否稳定。

产出 reports/daily_study_<date>.md + 两张图。证伪优先：关系弱就如实写弱。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MPL_FONT, QUINTILES, REPORTS_DIR
from data import store
from research.overnight import build_daily_features

plt.rcParams["font.sans-serif"] = [MPL_FONT]
plt.rcParams["axes.unicode_minus"] = False


def _ols(x, y) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(np.asarray(y, float), sm.add_constant(np.asarray(x, float))).fit()


def run() -> str:
    sse_daily = store.load("sse_daily")
    xau_daily = store.load("xau_daily")
    if sse_daily is None or xau_daily is None:
        raise RuntimeError("本地缺少日线数据，请先运行 scripts/sync_daily.py")

    df = build_daily_features(sse_daily, xau_daily).dropna(
        subset=["gold_ret_prev", "sse_intraday"]).reset_index(drop=True)
    n_total = len(df)

    # --- 全样本 OLS（滞后一日金收益 → 次日上证日内收益） ---
    res_prev = _ols(df["gold_ret_prev"], df["sse_intraday"])
    res_same = _ols(df["gold_ret_same"].dropna(),
                    df.loc[df["gold_ret_same"].notna(), "sse_intraday"])
    corr_prev = df[["gold_ret_prev", "sse_intraday"]].corr().iloc[0, 1]
    corr_same = df[["gold_ret_same", "sse_intraday"]].corr().iloc[0, 1]

    # --- 逐年相关性 / 样本数 ---
    by_year = (
        df.groupby("year")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "corr": g["gold_ret_prev"].corr(g["sse_intraday"]),
            "beta": _ols(g["gold_ret_prev"], g["sse_intraday"]).params[1]
            if len(g) > 10 else np.nan,
            "sse_intraday_mean": g["sse_intraday"].mean(),
        }), include_groups=False)
        .reset_index()
    )
    n_year_pos = int((by_year["corr"] > 0).sum())
    n_year_neg = int((by_year["corr"] < 0).sum())

    # --- 按金收益分位数的条件收益 ---
    df["gold_q"] = pd.qcut(df["gold_ret_prev"], QUINTILES,
                           labels=[f"Q{i+1}" for i in range(QUINTILES)])
    quint = df.groupby("gold_q", observed=True)["sse_intraday"].agg(
        ["count", "mean", "std"])
    hit_map = df.assign(sign=np.sign(df["gold_ret_prev"])).groupby("sign")[
        "sse_intraday"].mean()

    # --- regime：按金 20 日波动率三分位 ---
    df["vol_r"] = pd.qcut(df["gold_vol20_prev"], 3,
                          labels=["低波动", "中波动", "高波动"])
    regime = df.groupby("vol_r", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "corr": g["gold_ret_prev"].corr(g["sse_intraday"]),
            "cond_mean": np.sign(g["gold_ret_prev"]).mul(g["sse_intraday"]).mean(),
        }), include_groups=False).reset_index()

    # --- 滚动 252 日相关性 ---
    roll = df.set_index("date")["gold_ret_prev"].rolling(252).corr(
        df.set_index("date")["sse_intraday"])

    # --- 图 ---
    fig, ax = plt.subplots(figsize=(10, 4))
    roll.plot(ax=ax)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("滚动252日相关性：伦敦金日收益(滞后) vs 上证日内收益")
    ax.set_ylabel("Pearson 相关性")
    fig.tight_layout()
    p1 = REPORTS_DIR / "daily_study_rolling_corr.png"
    fig.savefig(p1, dpi=120)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in quint["mean"]]
    quint["mean"].plot(kind="bar", ax=ax, color=colors)
    ax.set_title("按伦敦金收益分位的次日上证日内平均收益")
    ax.set_xlabel("伦敦金收益分位（Q1最低 → Q5最高）")
    ax.set_ylabel("上证日内平均收益")
    fig.tight_layout()
    p2 = REPORTS_DIR / "daily_study_quintile.png"
    fig.savefig(p2, dpi=120)

    # --- 报告 ---
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    lines = [
        f"# 黄金-上证日线关系研究（{today}）",
        "",
        f"- 样本：{df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}，共 {n_total} 个交易日",
        f"- 特征口径：**滞后一日**伦敦金 close-to-close 收益（日线代理，非精确隔夜窗；精确口径见分钟回测）",
        "",
        "## 全样本",
        "",
        f"- 相关性（滞后金收益 vs 上证日内收益）：**{corr_prev:.4f}**",
        f"- OLS：beta = {res_prev.params[1]:+.4f}（t = {res_prev.tvalues[1]:+.2f}），R² = {res_prev.rsquared:.4f}",
        f"- 对照（同日金收益，含盘后信息，仅看大小）：corr = {corr_same:.4f}",
        "",
        "## 分位数条件收益（滞后金收益 5 分位 → 次日上证日内平均收益）",
        "",
        quint.to_markdown(floatfmt=".5f"),
        "",
        f"- 金涨日 vs 金跌日之后的上证日内平均收益：金涨日 {hit_map.get(1.0, np.nan):+.5f} / 金跌日 {hit_map.get(-1.0, np.nan):+.5f}",
        "",
        "## 逐年相关性",
        "",
        by_year.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"- 相关性为正的年份 {n_year_pos} 个 / 为负 {n_year_neg} 个"
        f"（{n_year_pos + n_year_neg} 个有效年份，符号一致性"
        f" {max(n_year_pos, n_year_neg) / max(1, n_year_pos + n_year_neg):.0%}）",
        "",
        "## 波动率 regime",
        "",
        regime.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 图",
        "",
        f"![滚动相关性](daily_study_rolling_corr.png)",
        f"![分位数条件收益](daily_study_quintile.png)",
        "",
        "## 诚实地说明",
        "",
        "- 日线口径是**近似**：日线金收益覆盖整个 24h，不是精确的「上证 15:00 → 09:30」隔夜窗口；"
        "精确口径在分钟数据积累后由 minute_backtest 给出",
        "- 若 |corr| 长期 < 0.05，说明该信号强度不足以独立交易，"
        "建议加入对照变量（USD/CNH、富时A50期货、隔夜美股）或直接放弃",
        "- 本报告不构成投资建议",
    ]
    out = REPORTS_DIR / f"daily_study_{pd.Timestamp.now():%Y%m%d}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[study] 报告已生成: {out}")
    print(f"[study] 全样本 corr={corr_prev:+.4f}, beta t值={res_prev.tvalues[1]:+.2f}, R²={res_prev.rsquared:.4f}")
    return str(out)
