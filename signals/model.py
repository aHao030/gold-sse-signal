"""滚动 OLS：上证日内收益 ~ 隔夜伦敦金收益。"""
from __future__ import annotations

import numpy as np
import statsmodels.api as sm


class OvernightGoldModel:
    """普通最小二乘，min_train 个样本前不输出信号；window 为滚动窗口（None=扩展窗口）。"""

    def __init__(self, min_train: int = 20, window: int | None = None):
        self.min_train = min_train
        self.window = window
        self.res_ = None
        self.n_ = 0

    def fit(self, x, y) -> "OvernightGoldModel":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        self.n_ = int(len(x))
        if self.n_ < self.min_train:
            self.res_ = None
            return self
        if self.window:
            x, y = x[-self.window:], y[-self.window:]
        self.res_ = sm.OLS(y, sm.add_constant(x)).fit()
        return self

    @property
    def ready(self) -> bool:
        return self.res_ is not None

    def predict(self, x: float) -> float:
        return float(self.res_.predict([1.0, float(x)])[0])

    def predict_with_ci(self, x: float, alpha: float = 0.05) -> tuple[float, float, float]:
        """返回 (预测值, 置信下界, 置信上界)——均值响应的置信区间。"""
        pred = self.res_.get_prediction([1.0, float(x)])
        ci = pred.conf_int(alpha=alpha)[0]
        return float(pred.predicted_mean[0]), float(ci[0]), float(ci[1])

    def describe(self) -> dict:
        if not self.ready:
            return {"n": self.n_, "ready": False}
        return {
            "n": self.n_,
            "ready": True,
            "alpha": float(self.res_.params[0]),
            "beta": float(self.res_.params[1]),
            "t_beta": float(self.res_.tvalues[1]),
            "r2": float(self.res_.rsquared),
        }
