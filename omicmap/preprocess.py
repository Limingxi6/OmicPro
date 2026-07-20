from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


EPS = 1e-8


@dataclass
class ModalityStats:
    """单模态的训练集拟合统计量（防泄漏）。"""

    columns: list[str] = field(default_factory=list)
    median: Optional[pd.Series] = None
    var_selected_columns: Optional[list[str]] = None
    mean: Optional[pd.Series] = None
    std: Optional[pd.Series] = None


class FoldPreprocessor:
    """
    每个 fold 内仅用训练集拟合，再应用到 valid/test。

    规则：
    - genotype: median 填补，保持 0/1/2 编码，不做 z-score
    - expression/metabolites: median 填补 + z-score
    - variance top-k: 在训练集上按方差筛选，并复用到 valid/test
    """

    def __init__(
        self,
        topk_genotype: Optional[int] = None,
        topk_expression: Optional[int] = None,
        topk_metabolites: Optional[int] = None,
    ) -> None:
        self.topk = {
            "genotype": topk_genotype,
            "expression": topk_expression,
            "metabolites": topk_metabolites,
        }
        self.stats: Dict[str, ModalityStats] = {
            "genotype": ModalityStats(),
            "expression": ModalityStats(),
            "metabolites": ModalityStats(),
        }
        self._is_fitted = False

    @staticmethod
    def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(pd.to_numeric, errors="coerce")

    @staticmethod
    def _select_topk_by_var(df: pd.DataFrame, k: Optional[int]) -> list[str]:
        cols = list(df.columns)
        if k is None or k <= 0 or k >= len(cols):
            return cols
        var = df.var(axis=0, ddof=0)
        top_cols = var.sort_values(ascending=False).head(k).index.tolist()
        return top_cols

    def _fit_one(self, name: str, x_train: pd.DataFrame, apply_zscore: bool) -> None:
        x_num = self._to_numeric_df(x_train)
        st = self.stats[name]
        st.columns = list(x_num.columns)
        st.median = x_num.median(axis=0)

        x_imp = x_num.fillna(st.median)
        selected_cols = self._select_topk_by_var(x_imp, self.topk[name])
        st.var_selected_columns = selected_cols
        x_sel = x_imp[selected_cols]

        if apply_zscore:
            st.mean = x_sel.mean(axis=0)
            st.std = x_sel.std(axis=0, ddof=0).replace(0.0, EPS)
        else:
            st.mean = None
            st.std = None

    def fit(
        self,
        x_train_geno: pd.DataFrame,
        x_train_expr: pd.DataFrame,
        x_train_metab: pd.DataFrame,
    ) -> "FoldPreprocessor":
        self._fit_one("genotype", x_train_geno, apply_zscore=False)
        self._fit_one("expression", x_train_expr, apply_zscore=True)
        self._fit_one("metabolites", x_train_metab, apply_zscore=True)
        self._is_fitted = True
        return self

    def _transform_one(self, name: str, x: pd.DataFrame, apply_zscore: bool) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("FoldPreprocessor 尚未 fit。")
        st = self.stats[name]
        if st.median is None or st.var_selected_columns is None:
            raise RuntimeError(f"{name} 统计量缺失。")

        x_num = self._to_numeric_df(x)
        # 对齐训练列，防止列缺失/顺序不一致
        x_num = x_num.reindex(columns=st.columns)
        x_imp = x_num.fillna(st.median)
        x_sel = x_imp[st.var_selected_columns]

        if apply_zscore:
            assert st.mean is not None and st.std is not None
            x_out = (x_sel - st.mean) / st.std
        else:
            x_out = x_sel
        return x_out

    def transform_genotype(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._transform_one("genotype", x, apply_zscore=False)

    def transform_expression(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._transform_one("expression", x, apply_zscore=True)

    def transform_metabolites(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._transform_one("metabolites", x, apply_zscore=True)

    def fit_transform_train(
        self,
        x_train_geno: pd.DataFrame,
        x_train_expr: pd.DataFrame,
        x_train_metab: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        self.fit(x_train_geno, x_train_expr, x_train_metab)
        return {
            "genotype": self.transform_genotype(x_train_geno),
            "expression": self.transform_expression(x_train_expr),
            "metabolites": self.transform_metabolites(x_train_metab),
        }

    @staticmethod
    def to_numpy_dict(
        x_geno: pd.DataFrame,
        x_expr: pd.DataFrame,
        x_metab: pd.DataFrame,
        y: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """转换为模型输入常用 ndarray。"""
        return {
            "geno": x_geno.to_numpy(dtype=np.float32),
            "expr": x_expr.to_numpy(dtype=np.float32),
            "metab": x_metab.to_numpy(dtype=np.float32),
            "y": y.to_numpy(dtype=np.float32),
        }
