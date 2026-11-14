"""全局配置：路径、交易时段、数据源符号、模型参数。"""
import os
from pathlib import Path

# 国内数据源直连（本机系统代理 127.0.0.1:7890 对东财/新浪偶发断连，实测直连稳定）；
# yahoo（yfinance）不走 NO_PROXY，继续走系统代理（实测直连 403）。
_NO_PROXY_DOMAINS = ",".join([
    "eastmoney.com", "sinajs.cn", "sina.com.cn", "sina.cn", "sinaurl.cn",
    "sge.com.cn",
])
os.environ["NO_PROXY"] = _NO_PROXY_DOMAINS
os.environ["no_proxy"] = _NO_PROXY_DOMAINS

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data_store"   # 本地数据落盘（gitignore，不上传）
PARQUET_DIR = DATA_DIR / "parquet"
LOGS_DIR = DATA_DIR / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"   # 研究报告与图（进 git）

# --- 标的与数据源 ---
SSE_INDEX = "000001"          # 上证指数（东财 secid: 1.000001）
GOLD_DAILY_SYMBOL = "XAU"     # 新浪 伦敦金现日线（akshare futures_foreign_hist，2006-11 至今）
GOLD_MINUTE_SYMBOL = "GC=F"   # COMEX 黄金主力连续（yfinance，5m/60天、1m/30天）
GOLD_EM_SECID = "101.GC00Y"   # 东财 COMEX 金主力（分钟备用源，klt=5/60）
TZ = "Asia/Shanghai"          # 所有时间戳统一为北京时间

# --- A股交易时段（北京时间） ---
MORNING_OPEN = "09:30"
MORNING_CLOSE = "11:30"
AFTERNOON_OPEN = "13:00"
SSE_CLOSE = "15:00"           # 隔夜窗口起点：上一交易日 15:00 → 今日开盘
ENTRY_TIME = "09:31"          # 开盘后第一根 1 分钟 bar 收盘价进场
EXIT_TIME = "14:55"           # 尾盘平仓基准（收盘前卖出，不留隔夜仓）

# --- 模型参数 ---
MIN_TRAIN_DAYS = 20           # 最少训练样本（交易日），不足则不给信号
ROLLING_WINDOW_DAYS = 120     # 滚动 OLS 窗口（交易日）；None = 扩展窗口
QUINTILES = 5                 # 分位数分组数

# --- matplotlib 中文字体（Windows） ---
MPL_FONT = "Microsoft YaHei"

for _d in (PARQUET_DIR, LOGS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
