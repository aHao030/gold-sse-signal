# gold-sse-signal

基于「**隔夜伦敦金走势 → 上证指数盘中涨跌**」拟合关系的 A 股日内信号系统。

核心想法：伦敦金几乎 24 小时交易，而上证每日 09:30 开盘——开盘时，金已经走完了
上一交易日 15:00（上证收盘）以来的隔夜行情。本系统用这段**精确隔夜窗口的金收益**
做自变量，滚动 OLS 拟合它对上证**盘中涨跌程度**的预测力，给出当日日内方向/幅度信号。

**规则口径**：开盘（09:31 首根 bar 收盘价）跟踪信号 → 尾盘（14:55）平仓 → **不留隔夜仓**。
目前是**纯信号系统**，不下单。

## 目录结构

```
config.py               全局配置（时段/符号/模型参数）
data/                   数据层：sse.py(上证) gold.py(伦敦金) store.py(parquet增量) sync.py(每日同步)
research/               研究层：overnight.py(特征) daily_study.py(20年日线) minute_backtest.py(分钟回测)
signals/                信号层：model.py(滚动OLS) engine.py(每日信号)
scripts/                CLI：sync_daily.py / run_study.py / run_backtest.py / run_signal.py
reports/                研究报告与图（进 git）
data_store/             本地 parquet 与日志（不进 git）
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 数据同步（收盘后 15:10 之后跑；首次全量回填）
python scripts/sync_daily.py

# 2. 20 年日线关系研究（证伪优先：关系弱会直说）
python scripts/run_study.py

# 3. 分钟级 walk-forward 回测（开盘进/尾盘出）
python scripts/run_backtest.py --freq 1m

# 4. 每日开盘信号（09:25~09:31 运行）
python scripts/run_signal.py
python scripts/run_signal.py --dry-run   # 盘后用缓存验证
```

## 数据源与限制（重要）

| 数据 | 源 | 深度 | 说明 |
|---|---|---|---|
| 上证日线 | 东财 `index_zh_a_hist` | 1990 至今 | 免费无限制 |
| 上证 1 分钟 | 东财 trends2 | **仅最近约 5 个交易日**（滚动） | 每日收盘后同步才能积累 |
| 上证 5 分钟 | 东财 kline | 约 36 个交易日（滚动） | 同上 |
| 伦敦金日线 | 新浪 `futures_foreign_hist("XAU")` | 2006-11 至今 | 节假日填充行已清洗 |
| 伦敦金 1 分钟 | yfinance `GC=F` | **最近 30 天**、单次请求 ≤7 天 | COMEX 金主力（23h 交易，覆盖隔夜段）；yahoo 在部分网络不可达，自动降级东财 `101.GC00Y` |
| 伦敦金 5 分钟 | yfinance `GC=F` | 最近 60 天 | 同上 |
| 伦敦金实时 | 新浪 `hf_XAU` | 秒级 | 必须带 Referer |

⚠️ 1 分钟样本在项目初期非常少（东财只留 5 天），统计结论有限；
**`sync_daily.py` 每天跑一次，数据会按日积累**，这是本系统自建数据资产的核心动作。

## 诚实的风险声明

- 黄金与上证的相关性**预期偏弱且时变**（金更多反映避险/美元，A 股资金面另有驱动）。
  `run_study.py` 是证伪优先的：如果 |corr| 长期 < 0.05，报告会直接建议放弃或加入
  对照变量（USD/CNH、富时 A50 期货、隔夜美股）。
- 日线研究的特征是**近似口径**（金日线 close-to-close ≠ 精确的 15:00→09:30 隔夜窗）；
  精确口径以分钟数据积累后的 `minute_backtest` 为准。
- 免费数据源接口随时可能变动；信号系统不等于交易系统；本仓库一切内容不构成投资建议。

## 已知待验证项

- 东财 `101.GC00Y` 分钟线（klt=5/1）可用性——yfinance 不可达时的降级路径
- `hf_XAU` 实时串字段顺序——`run_signal.py --debug` 打印原始串核对
