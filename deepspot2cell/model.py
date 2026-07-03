from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import lightning as L
from typing import Union, Dict, Optional
from torchmetrics.functional import spearman_corrcoef
import math

from functools import partial
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from deepspot2cell.loss import (
    mse,
    safe_pearson,
    pearson_mse,
    huber,
    pearson_huber,
    spot_consistency_loss,
)
from deepspot2cell.utils.utils import fix_seed


class CellAttentionModule(nn.Module):
    """
    Multi-head attention that lets cells attend to each other's morphological features
    BEFORE aggregating into spot prediction.
    """

    def __init__(self, phi_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.phi_size = phi_size
        self.head_dim = phi_size // num_heads

        assert phi_size % num_heads == 0, "phi_size must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(phi_size, phi_size)
        self.k_proj = nn.Linear(phi_size, phi_size)
        self.v_proj = nn.Linear(phi_size, phi_size)

        # Output projection
        self.out_proj = nn.Linear(phi_size, phi_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, phi_cells, mask):
        """
        Args:
            phi_cells: [Batch, Cells, phi_size] - Cell morphology features
            mask: [Batch, Cells] - 1 for valid cells, 0 for padding
        Returns:
            attended_cells: [Batch, Cells, phi_size] - Context-aware cell features
        """
        B, C, D = phi_cells.shape

        # Project to Q, K, V
        Q = self.q_proj(phi_cells)  # [B, C, D]
        K = self.k_proj(phi_cells)
        V = self.v_proj(phi_cells)

        # Reshape for multi-head attention
        Q = Q.view(B, C, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, C, d]
        K = K.view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, C, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # [B, H, C, C]

        # Mask padded cells (both in queries AND keys)
        # mask: [B, C] -> [B, 1, 1, C] for broadcasting
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, C]
        scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        # Also mask queries (rows) - if a cell is padding, it shouldn't attend to anything
        query_mask = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, C, 1]
        scores = scores.masked_fill(query_mask == 0, float("-inf"))

        # Softmax + handle NaNs (spots with no valid cells)
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, C, C]
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [B, H, C, d]

        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(B, C, D)  # [B, C, D]

        # Output projection
        output = self.out_proj(attended)

        return output


class Phi(nn.Module):
    def __init__(self, input_size: int, output_size: int, p: float = 0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Dropout(p=p),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, output_size),
            nn.Dropout(p=p),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Rho(nn.Module):
    def __init__(self, input_size: int, output_size: int, p: float = 0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, output_size),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GeneProjectionHead(nn.Module):
    """Bilinear gene-conditioned output head.

    Projects cell state and gene embeddings into a shared space, then scores each
    (cell, gene) pair via dot product. Replaces the flat Rho output head when
    gene embeddings are provided, enabling panel-agnostic inference.
    """

    def __init__(self, cell_dim: int, gene_emb_dim: int, proj_dim: int, p: float = 0.0):
        super().__init__()
        self.cell_proj = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(cell_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.gene_proj = nn.Linear(gene_emb_dim, proj_dim, bias=False)
        self.gene_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, cell_state: torch.Tensor, gene_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        cell_state: [..., cell_dim]
        gene_embs:  [N_genes, gene_emb_dim]
        returns:    [..., N_genes], Softplus activated
        """
        cell_proj = self.cell_proj(cell_state)  # [..., proj_dim]
        gene_proj = self.gene_proj(gene_embs)  # [N_genes, proj_dim]
        return F.softplus(cell_proj @ gene_proj.t() + self.gene_bias)


class CellPositionalEncoding(nn.Module):
    """Learns a spatial embedding from normalized (x, y) cell coordinates within a patch."""

    def __init__(self, output_size: int, hidden_size: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.norm = nn.LayerNorm(output_size)

    def forward(self, coords):
        """
        Args:
            coords: [Batch, Cells, 2] - normalized (x, y) in [0, 1]
        Returns:
            [Batch, Cells, output_size] - positional embeddings
        """
        return self.norm(self.model(coords))


class DeepSpot2Cell(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        lr: float = 1e-4,
        p: float = 0.3,
        p_phi: Union[None, float] = None,
        p_rho: Union[None, float] = None,
        n_ensemble: int = 10,
        n_ensemble_phi: Union[None, int] = None,
        n_ensemble_rho: Union[None, int] = None,
        phi2rho_size: int = 512,
        weight_decay: float = 1e-6,
        random_seed: int = 2024,
        cell_gt_available=True,
        use_attention: bool = False,
        attention_heads: int = 4,
        loss_type: str = "mse",
        huber_delta: float = 1.0,
        context_dropout: float = 0.0,
        use_positional_encoding: bool = False,
        aux_loss_weight: float = 0.0,
        pooling: str = "sum",
        pool_layer_norm: bool = False,
        predict_then_sum: bool = False,
        input_layer_norm: bool = False,
        warmup_epochs: int = 0,
        cosine_epochs: int = 0,
        eta_min_factor: float = 0.01,
        gene_emb_dim: int = 0,
        gene_proj_dim: int = 256,
        gene_embs: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["gene_embs"])
        fix_seed(random_seed)

        self.model_name = f"deepspot2cell"
        self.p_phi = p_phi or p
        self.p_rho = p_rho or p
        self.n_ensemble_phi = n_ensemble_phi or n_ensemble
        self.n_ensemble_rho = n_ensemble_rho or n_ensemble

        self.context_dropout_layer = nn.Dropout(p=context_dropout)
        if context_dropout > 0.0:
            print(f"[Info] Using Context Dropout with p={context_dropout}")

        # Get loss type
        self.loss_type = loss_type.lower()
        if self.loss_type == "pearson_mse":
            self.loss_fn = pearson_mse
            print("[Info] Using Hybrid Loss: Pearson + MSE")
        elif self.loss_type == "mse":
            self.loss_fn = mse
            print("[Info] Using Standard Loss: MSE")
        elif self.loss_type == "huber":
            self.loss_fn = partial(huber, delta=huber_delta)
            print(f"[Info] Using Huber Loss (delta={huber_delta})")
        elif self.loss_type == "pearson_huber":
            self.loss_fn = partial(pearson_huber, delta=huber_delta)
            print(f"[Info] Using Hybrid Loss: Pearson + Huber (delta={huber_delta})")
        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}. Supported: 'mse', 'pearson_mse', 'huber', 'pearson_huber'"
            )

        self.lr = lr
        self.weight_decay = weight_decay
        self.use_attention = use_attention

        self.train_batch_losses = []
        self.train_cell_inside_preds = []
        self.train_cell_inside_targets = []
        self.train_cell_outside_preds = []
        self.train_cell_outside_targets = []

        self.val_batch_losses = []
        self.val_cell_inside_preds = []
        self.val_cell_inside_targets = []
        self.val_cell_outside_preds = []
        self.val_cell_outside_targets = []

        self.val_spot_preds = []
        self.val_spot_targets = []

        mult = 3

        self.attention_heads = attention_heads
        if self.use_attention:
            # Cell-to-cell attention (applied to phi_cell outputs)
            self.cell_attention = CellAttentionModule(
                phi_size=phi2rho_size, num_heads=attention_heads, dropout=p
            )
            print(f"[Info] Using Multi-Head Cell Attention ({attention_heads} heads)")
        else:
            self.cell_attention = None

        self.use_positional_encoding = use_positional_encoding
        if self.use_positional_encoding:
            self.cell_pos_encoder = CellPositionalEncoding(
                output_size=phi2rho_size, hidden_size=64
            )
            print(f"[Info] Using Cell Positional Encoding (2 -> {phi2rho_size})")
        else:
            self.cell_pos_encoder = None

        self.aux_loss_weight = self.hparams.get("aux_loss_weight", 0.0)
        if self.aux_loss_weight > 0.0:
            print(
                f"[Info] Using Auxiliary Spot Consistency Loss (weight={self.aux_loss_weight})"
            )

        # Pooling / aggregation flags
        self.predict_then_sum = predict_then_sum
        self.pooling = pooling.lower()
        self.pool_layer_norm = pool_layer_norm

        if input_layer_norm:
            self.input_norm = nn.LayerNorm(input_size)
            print(f"[Info] Using Input LayerNorm (dim={input_size})")
        else:
            self.input_norm = nn.Identity()

        if predict_then_sum:
            print(
                "[Info] Using Predict-then-Sum: rho applied per-cell in linear space, summed, then log1p"
            )
            if pooling != "sum" or pool_layer_norm:
                print(
                    "[Info] Note: 'pooling' and 'pool_layer_norm' are ignored when predict_then_sum=True"
                )
        else:
            if self.pooling == "mean":
                print("[Info] Using mean pooling instead of sum pooling")
            elif self.pooling != "sum":
                raise ValueError(
                    f"Unknown pooling type: {pooling}. Supported: 'sum', 'mean'"
                )
            if pool_layer_norm:
                print(
                    f"[Info] Using LayerNorm after pooling (dim={phi2rho_size * mult})"
                )
                self.pool_norm = nn.LayerNorm(phi2rho_size * mult)

        self.phi_spot = nn.ModuleList(
            [
                Phi(input_size, phi2rho_size, self.p_phi)
                for _ in range(self.n_ensemble_phi)
            ]
        )
        self.phi_cell = nn.ModuleList(
            [
                Phi(input_size, phi2rho_size, self.p_phi)
                for _ in range(self.n_ensemble_phi)
            ]
        )

        self.use_gene_embs = gene_emb_dim > 0
        if self.use_gene_embs:
            _gene_embs = (
                gene_embs
                if gene_embs is not None
                else torch.zeros(output_size, gene_emb_dim)
            )
            self.register_buffer("gene_embs", _gene_embs)
            self.rho = nn.ModuleList(
                [
                    GeneProjectionHead(
                        phi2rho_size * mult, gene_emb_dim, gene_proj_dim, self.p_rho
                    )
                    for _ in range(self.n_ensemble_rho)
                ]
            )
            print(
                f"[Info] Using Gene Embedding Head (emb_dim={gene_emb_dim}, proj_dim={gene_proj_dim})"
            )
        else:
            self.rho = nn.ModuleList(
                [
                    Rho(phi2rho_size * mult, output_size, self.p_rho)
                    for _ in range(self.n_ensemble_rho)
                ]
            )

        self.cell_gt_available = cell_gt_available

    def forward(self, x, mask, context, context_mask, cell_centroids=None):
        return self._forward_superspot(x, mask, context, context_mask, cell_centroids)

    def _forward_single_cell(self, x, context, context_mask, cell_centroids=None):
        x = x.unsqueeze(0) if x.dim() == 1 else x
        n_cells, d_cell_emb = x.shape
        phi_cells = self._apply_phi_cell(x)

        if self.use_positional_encoding and cell_centroids is not None:
            pos_emb = self.cell_pos_encoder(cell_centroids)
            phi_cells = phi_cells + pos_emb

        if self.use_attention:
            # Need to add batch dimension for attention
            mask = torch.ones(n_cells, device=x.device)
            phi_cells = self.cell_attention(
                phi_cells.unsqueeze(0), mask.unsqueeze(0)
            ).squeeze(0)

        phi_context_spot_own = self._apply_phi(context[0].unsqueeze(0))

        neighbor_contexts = context[1:, :]
        neighbor_masks = context_mask[1:]
        d_ctx_emb = neighbor_contexts.shape[-1]
        phi_neighbor_contexts = self._apply_phi(
            neighbor_contexts.reshape(-1, d_ctx_emb)
        ).view(neighbor_contexts.shape[0], self.hparams.phi2rho_size)
        phi_neighbor_contexts_masked_for_sum = phi_neighbor_contexts.masked_fill(
            neighbor_masks.unsqueeze(-1), 0.0
        )
        sum_phi_neighbor_contexts = phi_neighbor_contexts_masked_for_sum.sum(
            dim=0, keepdim=True
        )
        num_valid_neighbors = (
            (~neighbor_masks).sum(dim=0, keepdim=True).float().clamp(min=1)
        )
        mean_phi_neighbor_contexts = sum_phi_neighbor_contexts / num_valid_neighbors

        phi_context_spot_own = self.context_dropout_layer(phi_context_spot_own)
        mean_phi_neighbor_contexts = self.context_dropout_layer(
            mean_phi_neighbor_contexts
        )

        phi_context_spot_own_expanded = phi_context_spot_own.expand(n_cells, -1)
        mean_phi_neighbor_contexts_expanded = mean_phi_neighbor_contexts.expand(
            n_cells, -1
        )

        concatenated_features = torch.cat(
            [
                phi_cells,
                phi_context_spot_own_expanded,
                mean_phi_neighbor_contexts_expanded,
            ],
            dim=1,
        )

        cell_counts = self._apply_rho(concatenated_features)
        return torch.log1p(cell_counts)

    def _forward_superspot(
        self,
        x,
        mask=None,
        context=None,
        context_mask=None,
        cell_centroids=None,
        return_cell_features=False,
    ):
        b, c, d_cell_emb = x.shape
        _b_ctx, n_total_ctx, d_ctx_emb = context.shape

        # 1. Extract cell features
        phi_cells = self._apply_phi_cell(x.view(-1, d_cell_emb)).view(b, c, -1)

        # 1b. Add positional encoding if enabled
        if self.use_positional_encoding and cell_centroids is not None:
            pos_emb = self.cell_pos_encoder(cell_centroids)
            phi_cells = phi_cells + pos_emb

        # 2. Apply attention to cell features BEFORE context concatenation
        if self.use_attention:
            phi_cells = self.cell_attention(phi_cells, mask)

        # 3. Extract context features (unchanged)
        phi_context_spot_own = self._apply_phi(context[:, 0, :])
        neighbor_contexts = context[:, 1:, :]
        neighbor_masks = context_mask[:, 1:]

        phi_neighbor_contexts_flat = self._apply_phi(
            neighbor_contexts.reshape(-1, d_ctx_emb)
        )
        phi_neighbor_contexts = phi_neighbor_contexts_flat.view(
            b, n_total_ctx - 1, self.hparams.phi2rho_size
        )
        phi_neighbor_contexts_masked = phi_neighbor_contexts.masked_fill(
            neighbor_masks.unsqueeze(-1), 0.0
        )
        sum_phi_neighbors = phi_neighbor_contexts_masked.sum(dim=1)
        num_valid_neighbors = (
            (~neighbor_masks).sum(dim=1, keepdim=True).float().clamp(min=1)
        )
        mean_phi_neighbors = sum_phi_neighbors / num_valid_neighbors

        phi_context_spot_own = self.context_dropout_layer(phi_context_spot_own)
        mean_phi_neighbors = self.context_dropout_layer(mean_phi_neighbors)

        # 4. Concatenate cell features with context
        phi_context_expanded = phi_context_spot_own.unsqueeze(1).expand(-1, c, -1)
        mean_phi_neighbors_expanded = mean_phi_neighbors.unsqueeze(1).expand(-1, c, -1)

        concatenated_per_cell = torch.cat(
            [phi_cells, phi_context_expanded, mean_phi_neighbors_expanded], dim=2
        )

        # 5 & 6. Aggregate + predict spot expression
        concatenated_per_cell_masked = concatenated_per_cell * mask.unsqueeze(-1)

        if self.predict_then_sum:
            # Apply rho per-cell in linear space, sum valid cells, then log1p.
            # This respects the physical additivity of transcript counts:
            #   log1p(sum_of_linear_cell_counts) vs incorrect log(A)+log(B)=log(A*B).
            flat_features = concatenated_per_cell_masked.view(b * c, -1)
            flat_counts = self._apply_rho(flat_features)  # [B*C, G], linear (Softplus)
            per_cell_counts = flat_counts.view(b, c, -1)  # [B, C, G]
            per_cell_counts_masked = per_cell_counts * mask.unsqueeze(-1)
            spot_pred = torch.log1p(per_cell_counts_masked.sum(dim=1))  # [B, G]
        else:
            # Standard DeepSets: pool features first, then apply rho once
            if self.pooling == "mean":
                n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1)
                aggregated_features = concatenated_per_cell_masked.sum(dim=1) / n_valid
            else:  # "sum"
                aggregated_features = concatenated_per_cell_masked.sum(dim=1)

            if self.pool_layer_norm:
                aggregated_features = self.pool_norm(aggregated_features)

            summed_counts = self._apply_rho(aggregated_features)
            spot_pred = torch.log1p(summed_counts)

        if return_cell_features:
            return spot_pred, concatenated_per_cell_masked
        return spot_pred

    def _apply_phi(self, x):
        return torch.stack([p(self.input_norm(x)) for p in self.phi_spot], 1).mean(1)

    def _apply_phi_cell(self, x):
        return torch.stack([p(self.input_norm(x)) for p in self.phi_cell], 1).mean(1)

    def _apply_rho(self, x):
        if self.use_gene_embs:
            return torch.stack([r(x, self.gene_embs) for r in self.rho], 1).mean(1)
        return torch.stack([r(x) for r in self.rho], 1).mean(1)

    def training_step(self, batch: Dict, batch_idx: int):
        cell_emb = batch["cell_embeddings"].float()
        spot_true = batch["spot_expression"].float()
        if self.cell_gt_available:
            cell_true = batch["cell_expressions"].float()
        mask = batch["cell_mask"].float()
        context = batch["neighb_embeddings"].float()
        context_mask = batch["neighb_masks"].bool()
        cell_centroids = batch["cell_centroids"].float()

        # pass only cells that are inside the spot: make outside cells zero
        is_inside = batch["is_inside_spot"].float()
        cell_emb_masked = cell_emb * is_inside.unsqueeze(-1)
        mask_inspot = mask * is_inside

        if self.aux_loss_weight > 0.0:
            pred, cell_features = self._forward_superspot(
                cell_emb_masked,
                mask_inspot,
                context,
                context_mask,
                cell_centroids=cell_centroids,
                return_cell_features=True,
            )
            main_loss = self.loss_fn(pred, spot_true)
            aux_loss = spot_consistency_loss(
                cell_features, self._apply_rho, mask_inspot, spot_true
            )
            loss = main_loss + self.aux_loss_weight * aux_loss
            self.log("train_aux_loss", aux_loss, on_epoch=True, prog_bar=False)
        else:
            pred = self(
                cell_emb_masked,
                mask_inspot,
                context,
                context_mask,
                cell_centroids=cell_centroids,
            )
            loss = self.loss_fn(pred, spot_true)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.train_batch_losses.append(loss.item())

        # Compute cell-level predictions for training metrics if ground truth available
        if self.cell_gt_available:
            with torch.no_grad():
                for b in range(cell_emb.size(0)):
                    valid = mask[b] > 0
                    if valid.any():
                        spot_context = context[b]
                        spot_context_mask = context_mask[b]

                        cell_preds = self._forward_single_cell(
                            cell_emb[b, valid],
                            spot_context,
                            spot_context_mask,
                            cell_centroids=cell_centroids[b, valid]
                            if self.use_positional_encoding
                            else None,
                        ).clamp_(min=0.0)
                        cell_targets = cell_true[b, valid]
                        is_inside_batch = is_inside[b, valid]

                        # Separate inside and outside cells
                        inside_mask = is_inside_batch > 0
                        outside_mask = is_inside_batch == 0

                        if inside_mask.any():
                            self.train_cell_inside_preds.append(
                                cell_preds[inside_mask].detach().cpu()
                            )
                            self.train_cell_inside_targets.append(
                                cell_targets[inside_mask].detach().cpu()
                            )

                        if outside_mask.any():
                            self.train_cell_outside_preds.append(
                                cell_preds[outside_mask].detach().cpu()
                            )
                            self.train_cell_outside_targets.append(
                                cell_targets[outside_mask].detach().cpu()
                            )

        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        cell_emb = batch["cell_embeddings"].float()
        spot_true = batch["spot_expression"].float()
        if self.cell_gt_available:
            cell_true = batch["cell_expressions"].float()
        mask = batch["cell_mask"].float()
        context = batch["neighb_embeddings"].float()
        context_mask = batch["neighb_masks"].bool()
        cell_centroids = batch["cell_centroids"].float()
        is_inside = batch["is_inside_spot"].float()
        mask_inspot = mask * is_inside  # mask out outside cells

        if self.aux_loss_weight > 0.0:
            spot_pred, cell_features = self._forward_superspot(
                cell_emb * is_inside.unsqueeze(-1),
                mask_inspot,
                context,
                context_mask,
                cell_centroids=cell_centroids,
                return_cell_features=True,
            )
            spot_pred = spot_pred.clamp_(min=0.0)
            main_loss = self.loss_fn(spot_pred, spot_true)
            aux_loss = spot_consistency_loss(
                cell_features, self._apply_rho, mask_inspot, spot_true
            )
            loss = main_loss + self.aux_loss_weight * aux_loss
            self.log("val_aux_loss", aux_loss, on_epoch=True, prog_bar=False)
        else:
            spot_pred = self(
                cell_emb * is_inside.unsqueeze(-1),
                mask_inspot,
                context,
                context_mask,
                cell_centroids=cell_centroids,
            ).clamp_(min=0.0)
            loss = self.loss_fn(spot_pred, spot_true)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.val_batch_losses.append(loss.item())

        self.val_spot_preds.append(spot_pred.detach().cpu())
        self.val_spot_targets.append(spot_true.detach().cpu())

        if self.cell_gt_available:
            with torch.no_grad():
                for b in range(cell_emb.size(0)):
                    valid = mask[b] > 0
                    if valid.any():
                        spot_context = context[b]
                        spot_context_mask = context_mask[b]

                        cell_preds = self._forward_single_cell(
                            cell_emb[b, valid],
                            spot_context,
                            spot_context_mask,
                            cell_centroids=cell_centroids[b, valid]
                            if self.use_positional_encoding
                            else None,
                        ).clamp_(min=0.0)
                        cell_targets = cell_true[b, valid]
                        is_inside_batch = is_inside[b, valid]

                        # Separate inside and outside cells
                        inside_mask = is_inside_batch > 0
                        outside_mask = is_inside_batch == 0

                        if inside_mask.any():
                            self.val_cell_inside_preds.append(
                                cell_preds[inside_mask].detach().cpu()
                            )
                            self.val_cell_inside_targets.append(
                                cell_targets[inside_mask].detach().cpu()
                            )

                        if outside_mask.any():
                            self.val_cell_outside_preds.append(
                                cell_preds[outside_mask].detach().cpu()
                            )
                            self.val_cell_outside_targets.append(
                                cell_targets[outside_mask].detach().cpu()
                            )

        return loss

    def on_train_epoch_end(self):
        if self.cell_gt_available:
            # Compute cell pearson correlations for inside cells
            if self.train_cell_inside_preds:
                all_preds = torch.cat(self.train_cell_inside_preds, dim=0)
                all_targets = torch.cat(self.train_cell_inside_targets, dim=0)
                pearson = safe_pearson(all_preds, all_targets)
                self.log("train_cell_inside_pearson", pearson.mean(), prog_bar=True)

            # Compute cell pearson correlations for outside cells
            if self.train_cell_outside_preds:
                all_preds = torch.cat(self.train_cell_outside_preds, dim=0)
                all_targets = torch.cat(self.train_cell_outside_targets, dim=0)
                pearson = safe_pearson(all_preds, all_targets)
                self.log("train_cell_outside_pearson", pearson.mean(), prog_bar=True)

        self.train_cell_inside_preds.clear()
        self.train_cell_inside_targets.clear()
        self.train_cell_outside_preds.clear()
        self.train_cell_outside_targets.clear()

    def on_validation_epoch_end(self):
        # --- 1. COMPUTE SPOT-LEVEL METRICS (Pearson & Spearman) ---
        if self.val_spot_preds:
            all_spot_preds = torch.cat(self.val_spot_preds, dim=0)
            all_spot_targets = torch.cat(self.val_spot_targets, dim=0)

            # Pearson (Mean across genes)
            spot_pearson = safe_pearson(all_spot_preds, all_spot_targets)
            self.log("val_spot_pearson", spot_pearson.mean(), prog_bar=True)

            # Spearman (Mean across genes)
            # This calculates per-column (gene) Spearman correlation and averages them
            spot_spearman = spearman_corrcoef(all_spot_preds, all_spot_targets)
            spot_spearman_scalar = torch.nan_to_num(spot_spearman).mean()
            self.log("val_spot_spearman", spot_spearman_scalar, prog_bar=True)

        # Clear spot storage
        self.val_spot_preds.clear()
        self.val_spot_targets.clear()

        # --- 2. COMPUTE CELL-LEVEL METRICS (If GT available) ---
        if self.cell_gt_available:
            # Compute cell pearson correlations for inside cells
            if self.val_cell_inside_preds:
                all_preds = torch.cat(self.val_cell_inside_preds, dim=0)
                all_targets = torch.cat(self.val_cell_inside_targets, dim=0)
                pearson = safe_pearson(all_preds, all_targets)
                self.log("val_cell_inside_pearson", pearson.mean(), prog_bar=True)

            # Compute cell pearson correlations for outside cells
            if self.val_cell_outside_preds:
                all_preds = torch.cat(self.val_cell_outside_preds, dim=0)
                all_targets = torch.cat(self.val_cell_outside_targets, dim=0)
                pearson = safe_pearson(all_preds, all_targets)
                self.log("val_cell_outside_pearson", pearson.mean(), prog_bar=True)

        self.val_cell_inside_preds.clear()
        self.val_cell_inside_targets.clear()
        self.val_cell_outside_preds.clear()
        self.val_cell_outside_targets.clear()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        warmup = self.hparams.warmup_epochs
        cosine_end = self.hparams.cosine_epochs
        eta_min = self.lr * self.hparams.eta_min_factor

        if warmup == 0 and cosine_end == 0:
            return {"optimizer": optimizer}

        schedulers = []
        milestones = []

        if warmup > 0:
            schedulers.append(
                LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
                )
            )
            milestones.append(warmup)

        cosine_T = max(cosine_end - warmup, 1)
        schedulers.append(CosineAnnealingLR(optimizer, T_max=cosine_T, eta_min=eta_min))

        if len(schedulers) == 1:
            scheduler = schedulers[0]
        else:
            scheduler = SequentialLR(
                optimizer, schedulers=schedulers, milestones=milestones
            )

        print(
            f"[Info] LR schedule: {warmup} warmup epochs → cosine decay to epoch {cosine_end} (eta_min={eta_min:.2e})"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
