"""parquet 增量存储：按符号落盘、按主键去重、幂等可重跑。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import PARQUET_DIR


def _path(symbol: str) -> Path:
    return PARQUET_DIR / f"{symbol}.parquet"


def load(symbol: str) -> pd.DataFrame | None:
    p = _path(symbol)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def save(symbol: str, df: pd.DataFrame, key: str = "datetime") -> int:
    """合并写入并按主键去重（新数据优先），返回新增行数。"""
    if df is None or len(df) == 0:
        return 0
    df = df.copy()
    existing = load(symbol)
    n_before = len(existing) if existing is not None else 0
    if existing is not None:
        df = pd.concat([existing, df], ignore_index=True)
    df = (
        df.drop_duplicates(subset=[key], keep="last")
        .sort_values(key)
        .reset_index(drop=True)
    )
    df.to_parquet(_path(symbol), index=False)
    return len(df) - n_before


def date_range(symbol: str, key: str = "datetime") -> tuple | None:
    df = load(symbol)
    if df is None or len(df) == 0:
        return None
    return df[key].min(), df[key].max()
