from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data_utils import AlignedData, get_fixed_fold_split, load_and_align_data
from .preprocess import FoldPreprocessor


class MultiModalDataset(Dataset):
    """返回 (geno, expr, metab, y, sample_id) 的多模态 Dataset。"""

    def __init__(
        self,
        sample_ids: list[str],
        x_geno: np.ndarray,
        x_expr: np.ndarray,
        x_metab: np.ndarray,
        y: np.ndarray,
    ) -> None:
        assert len(sample_ids) == len(x_geno) == len(x_expr) == len(x_metab) == len(y)
        self.sample_ids = sample_ids
        self.x_geno = torch.as_tensor(x_geno, dtype=torch.float32)
        self.x_expr = torch.as_tensor(x_expr, dtype=torch.float32)
        self.x_metab = torch.as_tensor(x_metab, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        return {
            "sample_id": self.sample_ids[idx],
            "geno": self.x_geno[idx],
            "expr": self.x_expr[idx],
            "metab": self.x_metab[idx],
            "y": self.y[idx],
        }


@dataclass
class FoldDataBundle:
    """单个 fold 的 Dataset/DataLoader 以及预处理器。"""

    train_dataset: MultiModalDataset
    valid_dataset: MultiModalDataset
    test_dataset: MultiModalDataset
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader
    preprocessor: FoldPreprocessor
    split_ids: Dict[str, list[str]]


def _build_dataset(
    aligned: AlignedData,
    ids: list[str],
    preprocessor: FoldPreprocessor,
) -> MultiModalDataset:
    g = preprocessor.transform_genotype(aligned.genotype.loc[ids])
    e = preprocessor.transform_expression(aligned.expression.loc[ids])
    m = preprocessor.transform_metabolites(aligned.metabolites.loc[ids])
    y = aligned.phenotype.loc[ids]

    return MultiModalDataset(
        sample_ids=ids,
        x_geno=g.to_numpy(dtype=np.float32),
        x_expr=e.to_numpy(dtype=np.float32),
        x_metab=m.to_numpy(dtype=np.float32),
        y=y.to_numpy(dtype=np.float32),
    )


def build_fold_dataloaders(
    data_dir: str | Path,
    test_fold: int,
    valid_fold: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    topk_genotype: Optional[int] = None,
    topk_expression: Optional[int] = None,
    topk_metabolites: Optional[int] = None,
    pin_memory: bool = False,
) -> FoldDataBundle:
    """
    构建固定十折下单个 fold 的 Dataset/DataLoader。

    关键防泄漏逻辑：
    - 先按固定 fold 切 train/valid/test
    - 只用 train 拟合 imputing / variance filtering / z-score
    - 再应用到 valid/test
    """
    aligned = load_and_align_data(data_dir=data_dir)
    split_ids = get_fixed_fold_split(aligned.fold_map, test_fold=test_fold, valid_fold=valid_fold)

    preprocessor = FoldPreprocessor(
        topk_genotype=topk_genotype,
        topk_expression=topk_expression,
        topk_metabolites=topk_metabolites,
    )

    # 仅训练集拟合
    train_ids = split_ids["train"]
    preprocessor.fit(
        aligned.genotype.loc[train_ids],
        aligned.expression.loc[train_ids],
        aligned.metabolites.loc[train_ids],
    )

    train_ds = _build_dataset(aligned, split_ids["train"], preprocessor)
    valid_ds = _build_dataset(aligned, split_ids["valid"], preprocessor)
    test_ds = _build_dataset(aligned, split_ids["test"], preprocessor)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return FoldDataBundle(
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        test_dataset=test_ds,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        preprocessor=preprocessor,
        split_ids=split_ids,
    )
