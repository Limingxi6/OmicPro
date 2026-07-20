from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_net import MambaOmicsExtractor


def _build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    dropout: float = 0.2,
    use_batchnorm: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _bitmask_to_present_mask(missing_code: torch.Tensor) -> torch.Tensor:
    code = missing_code.to(dtype=torch.long)
    g_present = ((code & 1) == 0).to(dtype=torch.float32)
    e_present = ((code & 2) == 0).to(dtype=torch.float32)
    m_present = ((code & 4) == 0).to(dtype=torch.float32)
    return torch.stack([g_present, e_present, m_present], dim=1)


def _gate_init_to_logit(gate_init: float) -> float:
    v = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
    return math.log(v / (1.0 - v))


def _to_gate_vector(value: float | Sequence[float], dim: int, name: str) -> torch.Tensor:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        vals = [float(v) for v in value]
    else:
        vals = [float(value)]
    if len(vals) == 1:
        vals = vals * dim
    if len(vals) != dim:
        raise ValueError(f"{name} expects scalar or length-{dim} list, got length={len(vals)}")
    return torch.tensor(vals, dtype=torch.float32)


class ResidualSummaryAttention(nn.Module):
    """
    Multi-head attention to aggregate a short list of modality summaries.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim({self.dim}) must be divisible by num_heads({self.num_heads})")
        self.head_dim = self.dim // self.num_heads

        self.key_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.val_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.query = nn.Parameter(torch.randn(self.num_heads, self.head_dim) * 0.02)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_norm = nn.LayerNorm(self.dim)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, history: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(history) == 0:
            raise ValueError("history must be non-empty")

        h = torch.stack(history, dim=1)
        hk = self.key_proj(F.rms_norm(h, normalized_shape=(self.dim,)))
        hv = self.val_proj(h)

        hk = hk.view(h.size(0), h.size(1), self.num_heads, self.head_dim)
        hv = hv.view(h.size(0), h.size(1), self.num_heads, self.head_dim)

        scores = torch.einsum("blhd,hd->bhl", hk, self.query) / math.sqrt(float(self.head_dim))
        weights = torch.softmax(scores, dim=-1)
        mixed_heads = torch.einsum("bhl,blhd->bhd", weights, hv)
        mixed = mixed_heads.reshape(h.size(0), self.dim)

        mixed = self.out_proj(mixed)
        mixed = self.out_norm(self.drop(mixed))
        return mixed, weights


class GenotypeCNNBranch(nn.Module):
    """
    Genotype branch (1D CNN).
    Input:  [B, 1, G]
    Output: [B, emb_dim]
    """

    def __init__(
        self,
        emb_dim: int = 128,
        conv_channels: Sequence[int] = (32, 64, 128),
        kernel_size: int = 5,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        padding = kernel_size // 2

        for out_ch in conv_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_ch, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.proj(x)
        return x


class OmicsMLPBranch(nn.Module):
    """
    MLP branch for expression or metabolites.
    Input:  [B, F]
    Output: [B, emb_dim]
    """

    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 128,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.net = _build_mlp(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=emb_dim,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MissingAwarePromptLearner(nn.Module):
    """
    Construct staged prompts per missing pattern.
    Layer-0 prompt length is 2 * prompt_length_half.
    Deeper-layer prompt length is prompt_length_half.
    """

    def __init__(
        self,
        embed_dim: int,
        prompt_length: int,
        prompt_depth: int,
        num_missing_types: int = 8,
    ) -> None:
        super().__init__()
        if prompt_length <= 0 or prompt_length % 3 != 0:
            raise ValueError("prompt_length must be positive and divisible by 3")
        if prompt_depth <= 0:
            raise ValueError("prompt_depth must be positive")

        self.embed_dim = int(embed_dim)
        self.prompt_depth = int(prompt_depth)
        self.prompt_length_half = int(prompt_length // 3)

        self.staged_prompt = nn.Parameter(
            torch.empty(num_missing_types, self.prompt_length_half, self.embed_dim)
        )
        self.common_prompt = nn.Parameter(
            torch.empty(num_missing_types, self.prompt_length_half, self.embed_dim)
        )
        nn.init.normal_(self.staged_prompt, std=0.02)
        nn.init.normal_(self.common_prompt, std=0.02)

        self.layernorm = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim * 2) for _ in range(max(0, self.prompt_depth - 1))]
        )
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embed_dim * 2, self.embed_dim),
                    nn.GELU(),
                    nn.Linear(self.embed_dim, self.embed_dim),
                )
                for _ in range(max(0, self.prompt_depth - 1))
            ]
        )

    def forward(self, missing_code: torch.Tensor) -> list[torch.Tensor]:
        code = missing_code.to(dtype=torch.long)
        staged = self.staged_prompt[code]  # [B, Lh, D]
        common = self.common_prompt[code]  # [B, Lh, D]

        all_prompts: list[torch.Tensor] = [torch.cat([staged, common], dim=1)]  # [B, 2*Lh, D]
        prev_staged = staged
        for depth in range(1, self.prompt_depth):
            corr_in = torch.cat([prev_staged, common], dim=-1)  # [B, Lh, 2D]
            corr = self.projections[depth - 1](self.layernorm[depth - 1](corr_in))
            all_prompts.append(corr)  # [B, Lh, D]
            prev_staged = corr

        return all_prompts


class PromptedResidualBlock(nn.Module):
    """
    DCP-aligned insertion order:
    1) Concatenate prompt tokens before attention.
    2) Apply self-attention residual.
    3) Apply MLP residual.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        prompt_length: int,
        layer_index: int,
        prompt_depth: int,
        num_missing_types: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.prompt_depth = int(prompt_depth)
        self.prompt_length = int(prompt_length)
        self.prompt_length_half = int(prompt_length // 3)
        self.first_layer = layer_index == 0

        self.attn = nn.MultiheadAttention(self.dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_1 = nn.LayerNorm(self.dim)
        self.ln_2 = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim * mlp_ratio, self.dim),
            nn.Dropout(dropout),
        )

        if layer_index == 0 and layer_index < self.prompt_depth:
            self.attn_prompt = nn.MultiheadAttention(
                self.dim,
                num_heads=1,
                dropout=0.0,
                batch_first=True,
            )
            self.prompts_dynamic = nn.Parameter(
                torch.empty(num_missing_types, self.prompt_length_half, self.dim)
            )
            nn.init.normal_(self.prompts_dynamic, std=0.02)
        else:
            self.attn_prompt = None
            self.prompts_dynamic = None

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln_1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        return y

    def forward(self, inputs: tuple[torch.Tensor, list[torch.Tensor], int, torch.Tensor]):
        x, compound_prompts_deeper, counter, missing_code = inputs
        if len(compound_prompts_deeper) > 0 and counter <= len(compound_prompts_deeper) - 1:
            if counter == 0:
                features = x if self.first_layer else x[:, self.prompt_length :, :]
                if self.attn_prompt is None or self.prompts_dynamic is None:
                    raise RuntimeError("Dynamic prompt module is missing for first prompt insertion.")
                prompts_dynamic_seed = self.prompts_dynamic[missing_code.to(dtype=torch.long)]
                prompts_dynamic, _ = self.attn_prompt(
                    prompts_dynamic_seed,
                    features,
                    features,
                    need_weights=False,
                )
                prompts_staged_and_common = compound_prompts_deeper[counter]
                x = torch.cat([prompts_staged_and_common, prompts_dynamic, features], dim=1)
                counter += 1
            else:
                features = x if self.first_layer else x[:, self.prompt_length :, :]
                prompts_dynamic_and_common = x[:, self.prompt_length_half : self.prompt_length_half * 3, :]
                prompts = compound_prompts_deeper[counter]
                x = torch.cat([prompts, prompts_dynamic_and_common, features], dim=1)
                counter += 1

        x = x + self._self_attention(x)
        x = x + self.mlp(self.ln_2(x))
        return x, compound_prompts_deeper, counter, missing_code


class PromptedFusionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        layers: int,
        num_heads: int,
        prompt_length: int,
        prompt_depth: int,
        num_missing_types: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                PromptedResidualBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    prompt_length=prompt_length,
                    layer_index=i,
                    prompt_depth=prompt_depth,
                    num_missing_types=num_missing_types,
                    dropout=dropout,
                )
                for i in range(layers)
            ]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        all_prompts: list[torch.Tensor],
        missing_code: torch.Tensor,
    ) -> torch.Tensor:
        state = (tokens, all_prompts, 0, missing_code.to(dtype=torch.long))
        for block in self.blocks:
            state = block(state)
        return state[0]


class MultiOmicsMultiTaskRegressor(nn.Module):
    """
    Missing-aware prompt model for 3 omics modalities.
    (Configured for single-target regression.)
    Prompt insertion keeps DCP's block-level ordering:
    prompt concat happens before attention in each prompted block.
    """

    def __init__(
        self,
        num_genotype_features: int,
        num_expression_features: int,
        num_metabolites_features: int,
        num_tasks: int = 1,
        branch_emb_dim: int = 128,
        fusion_hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        fusion_num_heads: int = 4,
        fusion_layers: int = 6,
        prompt_enable: bool = True,
        prompt_length: int = 12,
        prompt_depth: int = 3,
        num_missing_types: int = 8,
        prompt_gate_init: float | Sequence[float] = 0.05,
        prompt_gate_learnable: bool = True,
        prompt_gate_per_modality: bool = True,
        prompt_gate_use_missing_delta: bool = True,
        prompt_gate_missing_delta_scale: float = 0.2,
        prompt_gate_missing_delta_init_no_missing: float | Sequence[float] = -0.04,
        prompt_gate_missing_delta_init_missing: float | Sequence[float] = 0.12,
        prompt_disable_on_complete: bool = False,
        prompt_complete_update_scale: float = 1.0,
        mamba_layers: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ) -> None:
        super().__init__()

        self.num_genotype_features = num_genotype_features
        self.num_expression_features = num_expression_features
        self.num_metabolites_features = num_metabolites_features
        self.num_tasks = int(num_tasks)
        if self.num_tasks != 1:
            raise ValueError(f"Only single-regression mode is supported, got num_tasks={self.num_tasks}")

        self.prompt_enable = bool(prompt_enable)
        self.prompt_length = int(prompt_length)
        self.prompt_depth = int(prompt_depth)
        self.prompt_gate_learnable = bool(prompt_gate_learnable)
        self.prompt_gate_per_modality = bool(prompt_gate_per_modality)
        self.prompt_gate_dim = 3 if self.prompt_gate_per_modality else 1
        self.prompt_gate_use_missing_delta = bool(prompt_gate_use_missing_delta)
        self.prompt_gate_missing_delta_scale = float(max(0.0, prompt_gate_missing_delta_scale))
        self.num_missing_types = int(num_missing_types)
        self.prompt_disable_on_complete = bool(prompt_disable_on_complete)
        self.prompt_complete_update_scale = float(max(0.0, prompt_complete_update_scale))
        if self.prompt_length <= 0 or self.prompt_length % 3 != 0:
            raise ValueError("prompt_length must be positive and divisible by 3")

        self.genotype_branch = MambaOmicsExtractor(
            input_dim=num_genotype_features,
            emb_dim=branch_emb_dim,
            mamba_layers=mamba_layers,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        self.expression_branch = MambaOmicsExtractor(
            input_dim=num_expression_features,
            emb_dim=branch_emb_dim,
            mamba_layers=mamba_layers,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        self.metabolites_branch = MambaOmicsExtractor(
            input_dim=num_metabolites_features,
            emb_dim=branch_emb_dim,
            mamba_layers=mamba_layers,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        self.geno_aux_head = nn.Linear(branch_emb_dim, self.num_tasks)
        self.expr_aux_head = nn.Linear(branch_emb_dim, self.num_tasks)
        self.metab_aux_head = nn.Linear(branch_emb_dim, self.num_tasks)

        self.modality_attention = ResidualSummaryAttention(
            dim=branch_emb_dim,
            num_heads=4,
            dropout=0.05,
        )

        self.prompt_learner = MissingAwarePromptLearner(
            embed_dim=branch_emb_dim,
            prompt_length=self.prompt_length,
            prompt_depth=self.prompt_depth,
            num_missing_types=num_missing_types,
        )
        self.prompted_fusion = PromptedFusionTransformer(
            embed_dim=branch_emb_dim,
            layers=int(fusion_layers),
            num_heads=int(fusion_num_heads),
            prompt_length=self.prompt_length,
            prompt_depth=self.prompt_depth,
            num_missing_types=num_missing_types,
            dropout=dropout,
        )
        gate_init_vec = _to_gate_vector(prompt_gate_init, self.prompt_gate_dim, "prompt_gate_init").clamp(0.0, 1.0)
        if self.prompt_gate_learnable:
            self.prompt_gate_logit = nn.Parameter(
                torch.tensor([_gate_init_to_logit(float(v)) for v in gate_init_vec.tolist()], dtype=torch.float32)
            )
            self.register_buffer(
                "prompt_gate_const",
                torch.zeros(self.prompt_gate_dim, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.prompt_gate_logit = None
            self.register_buffer(
                "prompt_gate_const",
                gate_init_vec,
                persistent=False,
            )
        if self.prompt_gate_use_missing_delta:
            delta_no_missing = _to_gate_vector(
                prompt_gate_missing_delta_init_no_missing,
                self.prompt_gate_dim,
                "prompt_gate_missing_delta_init_no_missing",
            )
            delta_missing = _to_gate_vector(
                prompt_gate_missing_delta_init_missing,
                self.prompt_gate_dim,
                "prompt_gate_missing_delta_init_missing",
            )
            self.prompt_gate_missing_delta = nn.Embedding(self.num_missing_types, self.prompt_gate_dim)
            with torch.no_grad():
                self.prompt_gate_missing_delta.weight.copy_(
                    delta_missing.unsqueeze(0).repeat(self.num_missing_types, 1)
                )
                self.prompt_gate_missing_delta.weight[0, :] = delta_no_missing
        else:
            self.prompt_gate_missing_delta = None

        self.fusion_in_dim = branch_emb_dim * 3
        self.context_proj = nn.Linear(branch_emb_dim, self.fusion_in_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.fusion_in_dim, self.fusion_in_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(self.fusion_in_dim)

        self.fusion_head = _build_mlp(
            in_dim=self.fusion_in_dim,
            hidden_dims=fusion_hidden_dims,
            out_dim=self.num_tasks,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

    @staticmethod
    def _expand_gate_to_modalities(
        gate: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        g = gate.to(device=device, dtype=dtype)
        if g.dim() == 0:
            g = g.view(1, 1).repeat(batch_size, 1)
        elif g.dim() == 1:
            if g.numel() == 1:
                g = g.view(1, 1).repeat(batch_size, 1)
            elif g.numel() == batch_size:
                g = g.view(batch_size, 1)
            elif g.numel() == 3:
                g = g.view(1, 3).repeat(batch_size, 1)
            else:
                raise ValueError(f"Unsupported 1D gate length: {g.numel()}")
        elif g.dim() == 2:
            if g.size(0) == 1 and batch_size > 1:
                g = g.repeat(batch_size, 1)
            if g.size(0) != batch_size:
                raise ValueError(f"Gate batch mismatch: {g.size(0)} vs {batch_size}")
        else:
            raise ValueError(f"Unsupported gate tensor rank: {g.dim()}")

        if g.size(1) == 1:
            g = g.repeat(1, 3)
        elif g.size(1) != 3:
            raise ValueError(f"Gate channel mismatch: expected 1 or 3, got {g.size(1)}")
        return g

    def get_prompt_gate(self, missing_code: torch.Tensor | None = None) -> torch.Tensor:
        if self.prompt_gate_logit is not None:
            base_gate = torch.sigmoid(self.prompt_gate_logit)
        else:
            base_gate = self.prompt_gate_const

        if missing_code is None:
            return base_gate

        code = missing_code.to(dtype=torch.long).view(-1)
        if self.prompt_gate_missing_delta is None:
            return base_gate.unsqueeze(0).repeat(code.size(0), 1)

        delta = self.prompt_gate_missing_delta(code)
        if self.prompt_gate_missing_delta_scale > 0:
            delta = torch.tanh(delta) * self.prompt_gate_missing_delta_scale
        gate = base_gate.unsqueeze(0) + delta
        return gate.clamp(0.0, 1.0)

    def _compute_prompt_update_mask(
        self,
        present_mask: torch.Tensor,
        missing_code: torch.Tensor,
    ) -> torch.Tensor:
        present_mask = present_mask.to(dtype=torch.float32)
        missing_mask = (1.0 - present_mask).clamp(0.0, 1.0)
        if self.prompt_complete_update_scale <= 0.0:
            return missing_mask
        complete_mask = (missing_code.to(dtype=torch.long) == 0).to(dtype=torch.float32).view(-1, 1)
        complete_update = complete_mask * present_mask * self.prompt_complete_update_scale
        return (missing_mask + complete_update).clamp(0.0, 1.0)

    def get_effective_prompt_gate(
        self,
        missing_code: torch.Tensor,
        present_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        code = missing_code.to(dtype=torch.long).view(-1)
        if present_mask is None:
            present_mask = _bitmask_to_present_mask(code)
        else:
            present_mask = present_mask.to(dtype=torch.float32)

        gate = self.get_prompt_gate(code)
        gate = self._expand_gate_to_modalities(
            gate,
            batch_size=code.size(0),
            device=code.device,
            dtype=present_mask.dtype,
        )
        if self.prompt_disable_on_complete:
            gate = gate.masked_fill((code == 0).view(-1, 1), 0.0)

        update_mask = self._compute_prompt_update_mask(present_mask, code)
        effective = gate * update_mask
        if not self.prompt_enable:
            return torch.zeros_like(effective)
        return effective

    def forward(
        self,
        geno_x: torch.Tensor,
        expr_x: torch.Tensor,
        metab_x: torch.Tensor,
        missing_code: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = geno_x.device
        batch_size = geno_x.size(0)

        if missing_code is None:
            missing_code = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            missing_code = missing_code.to(device=device, dtype=torch.long)

        # 输入数据已经是缺失后的状态（缺失组学置零），直接提取 embedding
        g_emb = self.genotype_branch(geno_x)
        e_emb = self.expression_branch(expr_x)
        m_emb = self.metabolites_branch(metab_x)
        g_emb_extracted, e_emb_extracted, m_emb_extracted = g_emb, e_emb, m_emb

        use_prompt = self.prompt_enable
        if use_prompt and self.prompt_disable_on_complete and torch.all(missing_code == 0):
            use_prompt = False

        if use_prompt:
            tokens = torch.stack([g_emb, e_emb, m_emb], dim=1)  # [B, 3, D]
            base_tokens = tokens
            all_prompts = self.prompt_learner(missing_code)
            tokens = self.prompted_fusion(tokens, all_prompts, missing_code)
            tokens = tokens[:, self.prompt_length :, :]
            effective_gate = self.get_effective_prompt_gate(missing_code)
            effective_gate = effective_gate.to(dtype=tokens.dtype, device=tokens.device).unsqueeze(-1)
            tokens = base_tokens + effective_gate * (tokens - base_tokens)
            g_emb, e_emb, m_emb = tokens[:, 0, :], tokens[:, 1, :], tokens[:, 2, :]

        ctx, _ = self.modality_attention([g_emb, e_emb, m_emb])
        fused_base = torch.cat([g_emb, e_emb, m_emb], dim=1)
        fused = fused_base + self.fusion_gate(fused_base) * self.context_proj(ctx)
        fused = self.fusion_norm(fused)
        y_pred = self.fusion_head(fused)
        if not return_aux:
            return y_pred

        geno_aux = self.geno_aux_head(g_emb_extracted)
        expr_aux = self.expr_aux_head(e_emb_extracted)
        metab_aux = self.metab_aux_head(m_emb_extracted)
        return y_pred, geno_aux, expr_aux, metab_aux

