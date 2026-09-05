from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


TRAIT_COLUMNS = ["yd", "tp", "gn", "kgw"]


@dataclass
class AlignedData:
    """按 sample_id 对齐后的多模态数据。"""

    sample_ids: List[str]
    genotype: pd.DataFrame
    expression: pd.DataFrame
    metabolites: pd.DataFrame
    phenotype: pd.DataFrame
    fold_map: pd.Series  # index=sample_id, value=fold_id(int)


@dataclass
class AlignedModalities:
    """Three omics matrices aligned for phenotype-free prediction."""

    sample_ids: List[str]
    genotype: pd.DataFrame
    expression: pd.DataFrame
    metabolites: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file does not exist: {path}")
    errors: list[str] = []
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{enc}: {exc}")
    raise RuntimeError(f"Unable to parse CSV file {path}. Attempts: {' | '.join(errors)}")


def _detect_id_column(df: pd.DataFrame) -> str:
    normalized = {str(c).strip().lower(): str(c) for c in df.columns}
    for k in ("id", "sample_id", "sampleid"):
        if k in normalized:
            return normalized[k]
    return str(df.columns[0])


def _prepare_modality_df(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = df.copy()
    out[id_col] = out[id_col].astype(str).str.strip()
    if (out[id_col] == "").any():
        raise ValueError(f"Column '{id_col}' contains empty sample IDs.")
    duplicated = out.loc[out[id_col].duplicated(keep=False), id_col].unique().tolist()
    if duplicated:
        raise ValueError(
            f"Column '{id_col}' contains duplicate sample IDs; examples: {duplicated[:5]}"
        )
    out = out.set_index(id_col)
    return out


def _extract_fold_map(
    cvf_df: pd.DataFrame,
    sample_id_col: str,
    fold_column: Optional[str] = None,
) -> pd.Series:
    """
    解析 CVFs.csv 支持两类格式：
    1) sample_id + fold_id（如 ID, cv_1）
    2) sample_id + fold1...fold10（取值为0/1或one-hot风格）

    对于多列重复交叉验证（例如 cv_1...cv_10 每列都包含 1...10），
    必须通过 ``fold_column`` 明确选择一列，避免静默误读。
    """
    work = cvf_df.copy()
    work[sample_id_col] = work[sample_id_col].astype(str).str.strip()
    duplicated = work.loc[work[sample_id_col].duplicated(keep=False), sample_id_col].unique().tolist()
    if duplicated:
        raise ValueError(f"CV file contains duplicate sample IDs; examples: {duplicated[:5]}")

    non_id_cols = [c for c in work.columns if c != sample_id_col]
    if not non_id_cols:
        raise ValueError("CVFs.csv 不包含 fold 列。")

    if fold_column is not None:
        fold_col = str(fold_column)
        if fold_col not in non_id_cols:
            raise ValueError(
                f"CVFs 不包含指定的 fold_column={fold_col!r}；"
                f"可选列为: {list(map(str, non_id_cols))}"
            )
        fold_map = pd.to_numeric(work[fold_col], errors="coerce")
        if fold_map.isna().any():
            raise ValueError(f"CVFs 列 `{fold_col}` 含非数值 fold 值。")
        out = pd.Series(
            fold_map.astype(int).values,
            index=work[sample_id_col].values,
            name="fold_id",
        )
        return out

    # 格式1：只有一个非ID列，直接作为fold_id
    if len(non_id_cols) == 1:
        fold_col = non_id_cols[0]
        fold_map = pd.to_numeric(work[fold_col], errors="coerce")
        if fold_map.isna().any():
            raise ValueError(f"CVFs 列 `{fold_col}` 含非数值 fold 值。")
        out = pd.Series(fold_map.astype(int).values, index=work[sample_id_col].values, name="fold_id")
        return out

    # 格式2：多个 fold 列，只自动接受真正的 one-hot / 指示列。
    fold_like_cols = []
    for c in non_id_cols:
        lc = str(c).strip().lower()
        if re.search(r"(fold|cv)[_\-]?\d+$", lc):
            fold_like_cols.append(c)
    if not fold_like_cols:
        fold_like_cols = non_id_cols

    fold_block = work[fold_like_cols].apply(pd.to_numeric, errors="coerce")
    if fold_block.isna().all().all():
        raise ValueError("CVFs 多列格式无法解析为数值。")

    is_binary = fold_block.isin([0, 1]).all().all()
    is_one_hot = is_binary and (fold_block.sum(axis=1) == 1).all()
    if not is_one_hot:
        raise ValueError(
            "CVFs 含多个非 one-hot 列，可能是重复交叉验证划分。"
            "请在配置 data.fold_column 中明确选择一列，例如 cv_1。"
        )

    # 每行取最大值所在列作为fold
    max_col = fold_block.idxmax(axis=1)

    def _col_to_fold_id(name: str) -> int:
        m = re.search(r"(\d+)$", str(name))
        if m:
            return int(m.group(1))
        return fold_like_cols.index(name) + 1

    fold_id = max_col.map(_col_to_fold_id).astype(int)
    out = pd.Series(fold_id.values, index=work[sample_id_col].values, name="fold_id")
    return out


def load_and_align_data(
    data_dir: str | Path,
    geno_file: str = "Rice_geno_zhuanzhi.csv",
    expr_file: str = "Rice-Expression_zhaunzhi.csv",
    metab_file: str = "Rice_Metabolites_zhuanzhi.csv",
    pheno_file: str = "Rice-Phenotypes.csv",
    cvf_file: str = "CVFs.csv",
    fold_column: Optional[str] = None,
    traits: Optional[Sequence[str]] = None,
) -> AlignedData:
    """
    读取并按 sample_id 对齐 genotype/expression/metabolites/phenotype，
    只保留指定 phenotype 性状，并附带固定折号映射。
    """
    data_dir = Path(data_dir)
    traits = list(traits) if traits is not None else list(TRAIT_COLUMNS)

    geno = _read_csv(data_dir / geno_file)
    expr = _read_csv(data_dir / expr_file)
    metab = _read_csv(data_dir / metab_file)
    pheno = _read_csv(data_dir / pheno_file)
    cvf = _read_csv(data_dir / cvf_file)

    id_geno = _detect_id_column(geno)
    id_expr = _detect_id_column(expr)
    id_metab = _detect_id_column(metab)
    id_pheno = _detect_id_column(pheno)
    id_cvf = _detect_id_column(cvf)

    geno = _prepare_modality_df(geno, id_geno)
    expr = _prepare_modality_df(expr, id_expr)
    metab = _prepare_modality_df(metab, id_metab)
    pheno = _prepare_modality_df(pheno, id_pheno)

    missing_traits = [t for t in traits if t not in pheno.columns]
    if missing_traits:
        raise ValueError(f"Phenotype 缺少目标性状列: {missing_traits}")
    pheno = pheno[traits]

    fold_map = _extract_fold_map(cvf, id_cvf, fold_column=fold_column)

    # 仅保留五个来源共同出现样本
    common_ids = (
        set(geno.index.astype(str))
        & set(expr.index.astype(str))
        & set(metab.index.astype(str))
        & set(pheno.index.astype(str))
        & set(fold_map.index.astype(str))
    )
    if not common_ids:
        raise ValueError("五个文件没有共同 sample_id。")

    sample_ids = sorted(common_ids)
    geno = geno.loc[sample_ids]
    expr = expr.loc[sample_ids]
    metab = metab.loc[sample_ids]
    pheno = pheno.loc[sample_ids]
    fold_map = fold_map.loc[sample_ids]

    return AlignedData(
        sample_ids=sample_ids,
        genotype=geno,
        expression=expr,
        metabolites=metab,
        phenotype=pheno,
        fold_map=fold_map,
    )


def load_aligned_from_config(
    cfg: Mapping[str, Any],
    traits: Optional[Sequence[str]] = None,
) -> AlignedData:
    """Load a dataset using the explicit file mapping stored in a YAML config."""

    data_cfg = cfg.get("data", {})
    if data_cfg is None:
        data_cfg = {}
    if not isinstance(data_cfg, Mapping):
        raise TypeError("config.data must be a mapping of logical names to CSV files.")

    return load_and_align_data(
        data_dir=cfg.get("data_dir", "data"),
        geno_file=str(data_cfg.get("genotype", "Rice_geno_zhuanzhi.csv")),
        expr_file=str(data_cfg.get("expression", "Rice-Expression_zhaunzhi.csv")),
        metab_file=str(data_cfg.get("metabolites", "Rice_Metabolites_zhuanzhi.csv")),
        pheno_file=str(data_cfg.get("phenotype", "Rice-Phenotypes.csv")),
        cvf_file=str(data_cfg.get("folds", "CVFs.csv")),
        fold_column=(
            None
            if data_cfg.get("fold_column") in (None, "")
            else str(data_cfg.get("fold_column"))
        ),
        traits=traits,
    )


def load_prediction_modalities(
    data_dir: str | Path,
    geno_file: str = "Rice_geno_zhuanzhi.csv",
    expr_file: str = "Rice-Expression_zhaunzhi.csv",
    metab_file: str = "Rice_Metabolites_zhuanzhi.csv",
) -> AlignedModalities:
    """Load and align the three input modalities without requiring phenotype or CV files."""

    data_dir = Path(data_dir)
    frames: dict[str, pd.DataFrame] = {}
    for name, filename in (
        ("genotype", geno_file),
        ("expression", expr_file),
        ("metabolites", metab_file),
    ):
        frame = _read_csv(data_dir / filename)
        frames[name] = _prepare_modality_df(frame, _detect_id_column(frame))

    common_ids = (
        set(frames["genotype"].index.astype(str))
        & set(frames["expression"].index.astype(str))
        & set(frames["metabolites"].index.astype(str))
    )
    if not common_ids:
        raise ValueError("The three omics files do not share any sample IDs.")

    sample_ids = sorted(common_ids)
    return AlignedModalities(
        sample_ids=sample_ids,
        genotype=frames["genotype"].loc[sample_ids],
        expression=frames["expression"].loc[sample_ids],
        metabolites=frames["metabolites"].loc[sample_ids],
    )


def get_fixed_fold_split(
    fold_map: pd.Series,
    test_fold: int,
    valid_fold: Optional[int] = None,
) -> Dict[str, List[str]]:
    """
    基于固定十折映射产生 train/valid/test 样本ID。
    - test_fold: 作为测试集的fold
    - valid_fold: 若为空，默认取 (test_fold % 10) + 1
    """
    if valid_fold is None:
        valid_fold = (int(test_fold) % 10) + 1

    fold_int = pd.to_numeric(fold_map, errors="coerce").astype(int)
    ids = fold_map.index.astype(str)

    test_ids = ids[fold_int == int(test_fold)].tolist()
    valid_ids = ids[fold_int == int(valid_fold)].tolist()
    train_ids = ids[(fold_int != int(test_fold)) & (fold_int != int(valid_fold))].tolist()

    if len(test_ids) == 0:
        raise ValueError(f"test_fold={test_fold} 对应样本为空。")
    if len(valid_ids) == 0:
        raise ValueError(f"valid_fold={valid_fold} 对应样本为空。")
    if len(train_ids) == 0:
        raise ValueError("训练集为空，请检查 fold 划分。")

    return {"train": train_ids, "valid": valid_ids, "test": test_ids}
