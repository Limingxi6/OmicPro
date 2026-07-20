from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ModuleNotFoundError as exc:  # pragma: no cover
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


class SparseRegularizer(nn.Module):
    """
    MambaNet-style sparse reweighting:
    1) compute L1 norm over channel axis
    2) map to [0, 1] with sigmoid
    3) reweight input features
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"SparseRegularizer expects [B, C, F], got shape={tuple(x.shape)}")
        l1_norm = torch.norm(x, p=1, dim=1, keepdim=True)
        weights = torch.sigmoid(l1_norm)
        return x * weights


class StackedMambaBlocks(nn.Module):
    """
    Stacked Mamba blocks with global residual:
    x_out = x_in + Mamba_k(...Mamba_2(Mamba_1(x_in))...)
    """

    def __init__(
        self,
        num_layers: int = 2,
        d_model: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if Mamba is None:  # pragma: no cover
            raise RuntimeError(
                "mamba_ssm is required for MambaNet extractors. "
                "Install it with: pip install mamba-ssm"
            ) from _MAMBA_IMPORT_ERROR
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = x
        for layer in self.layers:
            out = layer(out)
        return residual + out


class MambaOmicsExtractor(nn.Module):
    """
    Per-omics feature extractor:
    sparse regularization -> stacked Mamba blocks -> projection to embedding
    """

    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        mamba_layers: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.sparse = SparseRegularizer()
        self.backbone = StackedMambaBlocks(
            num_layers=int(mamba_layers),
            d_model=1,
            d_state=int(mamba_d_state),
            d_conv=int(mamba_d_conv),
            expand=int(mamba_expand),
        )
        self.project = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim, int(emb_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"MambaOmicsExtractor expects [B, F] or [B, 1, F], got {tuple(x.shape)}")

        x = self.sparse(x)
        x = x.permute(0, 2, 1)  # [B, F, 1]
        x = self.backbone(x)
        x = x.permute(0, 2, 1).reshape(x.size(0), -1)  # [B, F]
        return self.project(x)
