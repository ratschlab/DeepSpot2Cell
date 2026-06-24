import argparse
import os
import sys
from pathlib import Path
import json
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import warnings
import gc
import psutil

from deepspot2cell import DeepSpot2Cell, DS2CDataset
from deepspot2cell.utils.utils import (
    fix_seed,
    clean_gene_string,
    load_xenium_gene_indices,
)

# Threshold below which predictions are zeroed for Spearman_Clipped
CLIP_THRESHOLD = 0.5

# Global Debug Flag
DEBUG_MODE = False

def print_system_status(tag=""):
    """Prints current RAM and VRAM usage."""
    if not DEBUG_MODE:
        return

    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2.0**30  # GB

    vram_str = "N/A"
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        vram_str = f"Alloc: {allocated:.2f}GB | Rsrv: {reserved:.2f}GB"

    print(f"[{tag}] RAM: {memory_use:.2f} GB | VRAM: {vram_str}")


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def check_data_exists(data_folder, dataset_variant, sample_ids):
    """Ensures H5 files exist before attempting inference."""
    missing = []
    for sid in sample_ids:
        path = (
            Path(data_folder)
            / f"expressions{dataset_variant}"
            / f"{sid}_expressions.h5"
        )
        if not path.exists():
            missing.append(str(path))

    if missing:
        print("FATAL: Missing preprocessed data files!")
        sys.exit(1)


def load_stats_from_file(stats_path, test_ds):
    if not stats_path.exists():
        print("Warning: Stats file not found. Assuming no scaling.")
        return
    
    print(f"Loading normalization statistics from {stats_path}...")
    stats = torch.load(stats_path, weights_only=False) 
    
    if stats.get("standard_scaling", False):
        test_ds.standard_scaling = True
        test_ds.spot_scaler = stats["spot_scaler"]
        test_ds.cell_scaler = stats["cell_scaler"]
    else:
        test_ds.standard_scaling = False

    if stats.get("minmax", False):
        test_ds.minmax = True
        test_ds.spot_gene_min = stats["spot_gene_min"]
        test_ds.spot_gene_max = stats["spot_gene_max"]
        test_ds.spot_gene_range = stats["spot_gene_range"]
        test_ds.cell_gene_min = stats.get("cell_gene_min", stats["spot_gene_min"])
        test_ds.cell_gene_max = stats.get("cell_gene_max", stats["spot_gene_max"])
        test_ds.cell_gene_range = stats.get("cell_gene_range", stats["spot_gene_range"])
    else:
        test_ds.minmax = False



def parse_global_coordinates(patch_barcodes, cell_centroids, valid_indices_mask):
    """Returns raw lists/arrays instead of a DataFrame to save RAM."""
    samples = []
    xs = []
    ys = []
    
    for b, barcode in enumerate(patch_barcodes):
        parts = barcode.split('_')
        row_idx = int(parts[1])
        col_idx = int(parts[2])
        sample_id = "_".join(parts[3:])
        
        mask = valid_indices_mask[b]
        if not mask.any():
            continue
            
        relative_centroids = cell_centroids[b, mask].cpu().numpy()
        
        # Global X = Column + Relative X
        # Global Y = Row + Relative Y
        global_x = col_idx + relative_centroids[:, 0]
        global_y = row_idx + relative_centroids[:, 1]
        
        samples.extend([sample_id] * len(global_x))
        xs.append(global_x)
        ys.append(global_y)
            
    return samples, xs, ys

def bootstrap_se(values, n_bootstraps=1000):
    if len(values) < 2: return 0.0
    values = np.array(values)
    boot_means = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    return np.std(boot_means)


def create_staggered_plot(sorted_df, buckets, title, filename, output_dir, color='blue', label='Mean Pearson', metric_col="Pearson"):
    """Updated to accept metric_col"""
    means, ses = [], []
    valid_buckets = [b for b in buckets if b <= len(sorted_df)]
    for k in valid_buckets:
        # CHANGED: Use metric_col instead of hardcoded "Pearson"
        vals = sorted_df.head(k)[metric_col].values 
        means.append(np.mean(vals))
        ses.append(bootstrap_se(vals))
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(range(len(valid_buckets)), means, yerr=ses, fmt='-o', capsize=5, color=color, label=label)
    plt.xticks(range(len(valid_buckets)), valid_buckets)
    plt.title(title)
    plt.xlabel("Top K Genes")
    plt.ylabel(metric_col) # CHANGED: Dynamic label
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / filename)
    plt.close()


def plot_spatial_genes_xenium(spatial_df, preds, targets, gene_names, top_genes_indices, output_dir):
    samples = spatial_df["Sample"].unique()
    samples_to_plot = samples[:8]
    
    for sample in samples_to_plot:
        sample_mask = spatial_df["Sample"] == sample
        sample_x = spatial_df.loc[sample_mask, "Global_X"].values
        sample_y = spatial_df.loc[sample_mask, "Global_Y"].values
        
        sample_preds = preds[sample_mask]
        sample_targets = targets[sample_mask]
        
        num_genes_to_plot = min(5, len(top_genes_indices))
        fig, axes = plt.subplots(num_genes_to_plot, 2, figsize=(12, 5 * num_genes_to_plot))
        fig.suptitle(f"Xenium Single-Cell Prediction vs Ground Truth: {sample}", fontsize=16)
        
        if num_genes_to_plot == 1: axes = np.array([axes])

        for i in range(num_genes_to_plot):
            gene_idx = top_genes_indices[i]
            gene_name = gene_names[gene_idx]
            
            p_vals = sample_preds[:, gene_idx]
            t_vals = sample_targets[:, gene_idx]
            
            try:
                p_c = p_vals - p_vals.mean()
                t_c = t_vals - t_vals.mean()
                denom = np.sqrt((p_c**2).sum() * (t_c**2).sum())
                r_val = float((p_c * t_c).sum() / denom) if denom > 0 else 0.0
            except Exception:
                r_val = 0.0

            vmin = min(np.percentile(p_vals, 1), np.percentile(t_vals, 1))
            vmax = max(np.percentile(p_vals, 99), np.percentile(t_vals, 99))
            
            sc1 = axes[i, 0].scatter(sample_x, -sample_y, c=p_vals, cmap="magma", s=1, vmin=vmin, vmax=vmax, alpha=0.8)
            axes[i, 0].set_title(f"Pred: {gene_name} (Sample R={r_val:.2f})")
            axes[i, 0].axis('off')
            plt.colorbar(sc1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            sc2 = axes[i, 1].scatter(sample_x, -sample_y, c=t_vals, cmap="magma", s=1, vmin=vmin, vmax=vmax, alpha=0.8)
            axes[i, 1].set_title(f"GT: {gene_name}")
            axes[i, 1].axis('off')
            plt.colorbar(sc2, ax=axes[i, 1], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = output_dir / f"xenium_spatial_plot_{sample}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()


def _pearson_per_gene(preds, targets):
    """Vectorized per-gene Pearson. preds/targets: (n_cells, n_genes). Returns (n_genes,)."""
    p = preds - preds.mean(axis=0)
    t = targets - targets.mean(axis=0)
    num = (p * t).sum(axis=0)
    denom = np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 0, num / denom, 0.0)
    return np.nan_to_num(r, nan=0.0)


def _spearman_per_gene(preds, targets):
    """Vectorized per-gene Spearman via rank-then-Pearson. preds/targets: (n_cells, n_genes)."""
    p_ranks = np.argsort(np.argsort(preds, axis=0), axis=0).astype(np.float32)
    t_ranks = np.argsort(np.argsort(targets, axis=0), axis=0).astype(np.float32)
    return _pearson_per_gene(p_ranks, t_ranks)


def print_per_sample_summary(final_preds, final_targets, spatial_df, results_df, valid_gene_names):
    """Print Pearson, Spearman, Spearman_Clipped, KNN-3/5 ceiling at @50/@100/@all + per-cell stats."""
    n_genes = len(valid_gene_names)
    fixed_buckets = [50, 100]
    gene_chunk = 500

    xs = spatial_df["Global_X"].values
    ys = spatial_df["Global_Y"].values
    samples_arr = spatial_df["Sample"].values

    metric_labels = [
        "Pearson", "Spearman", "Spear-Clip",
        "P-KNN3", "S-KNN3", "P-KNN5", "S-KNN5",
    ]

    SEP = "=" * 88
    DIV = "-" * 88
    bkt_hdr = "  ".join(f"@{b:>5}" for b in fixed_buckets) + f"  {'@all':>7}"
    print(f"\n{SEP}")
    print(f"  CEILING SUMMARY  —  Pearson | Spearman | Spearman_Clipped | KNN-3 | KNN-5")
    print(f"  Ceiling = genes sorted by own metric desc")
    print(SEP)
    print(f"  {'Sample':<18}  {'Metric':<12}  {bkt_hdr}")
    print(DIV)

    for sample_id in sorted(np.unique(samples_arr)):
        mask = samples_arr == sample_id
        indices = np.where(mask)[0]
        n_cells = len(indices)
        if n_cells < 10:
            print(f"  {sample_id:<18}  [skipped — {n_cells} cells]")
            continue

        fp = final_preds[indices].numpy()   # float32 tensor copy → numpy, no extra cast needed
        ft = final_targets[indices].numpy()
        fp_clipped = np.where(fp < CLIP_THRESHOLD, 0.0, fp).astype(np.float32)

        # Per-sample KNN smoothing (build tree once, query for both k values)
        coords = np.stack([xs[mask], ys[mask]], axis=1).astype(np.float32)
        tree = cKDTree(coords)
        smooth_t = {}
        for k in (3, 5):
            _, nn_idx = tree.query(coords, k=min(k + 1, n_cells))
            smooth_t[k] = ft[nn_idx].mean(axis=1)   # already float32

        # Per-gene metrics in chunks
        sample_metrics = {lbl: np.zeros(n_genes, dtype=np.float32) for lbl in metric_labels}
        for g0 in range(0, n_genes, gene_chunk):
            g1 = min(g0 + gene_chunk, n_genes)
            fp_c = fp[:, g0:g1]
            ft_c = ft[:, g0:g1]
            fpc_c = fp_clipped[:, g0:g1]
            sample_metrics["Pearson"][g0:g1]     = _pearson_per_gene(fp_c, ft_c)
            sample_metrics["Spearman"][g0:g1]    = _spearman_per_gene(fp_c, ft_c)
            sample_metrics["Spear-Clip"][g0:g1]  = _spearman_per_gene(fpc_c, ft_c)
            sample_metrics["P-KNN3"][g0:g1]      = _pearson_per_gene(fp_c, smooth_t[3][:, g0:g1])
            sample_metrics["S-KNN3"][g0:g1]      = _spearman_per_gene(fp_c, smooth_t[3][:, g0:g1])
            sample_metrics["P-KNN5"][g0:g1]      = _pearson_per_gene(fp_c, smooth_t[5][:, g0:g1])
            sample_metrics["S-KNN5"][g0:g1]      = _spearman_per_gene(fp_c, smooth_t[5][:, g0:g1])

        # Per-cell correlations (across genes, per cell)
        cell_pearson  = _pearson_per_gene(fp.T, ft.T)
        cell_spearman = _spearman_per_gene(fp.T, ft.T)

        for i, label in enumerate(metric_labels):
            vals = sample_metrics[label]
            order = np.argsort(vals)[::-1]
            parts = [f"{np.nanmean(vals[order[:b]]):>7.3f}" if b <= n_genes else "    ---"
                     for b in fixed_buckets]
            parts.append(f"{np.nanmean(vals):>7.3f}")
            row = "  ".join(parts)
            prefix = f"  {sample_id:<18}" if i == 0 else f"  {'':18}"
            n_str = f"  (n={n_cells:,})" if i == 0 else ""
            print(f"{prefix}  {label:<12}  {row}{n_str}")

        print(f"  {'':18}  {'cell-Pearson':12}  "
              f"mean={np.nanmean(cell_pearson):.4f}  "
              f"med={np.nanmedian(cell_pearson):.4f}  "
              f"std={np.nanstd(cell_pearson):.4f}")
        print(f"  {'':18}  {'cell-Spearman':12}  "
              f"mean={np.nanmean(cell_spearman):.4f}  "
              f"med={np.nanmedian(cell_spearman):.4f}  "
              f"std={np.nanstd(cell_spearman):.4f}")

        del fp, ft, fp_clipped, smooth_t, sample_metrics, cell_pearson, cell_spearman
        print(DIV)

    # Aggregate from results_df
    agg_cols = [
        ("Pearson",      "Pearson"),
        ("Spearman",     "Spearman"),
        ("Spear-Clip",   "Spearman_Clipped"),
        ("P-KNN3",       "Pearson_Smooth3"),
        ("S-KNN3",       "Spearman_Smooth3"),
        ("P-KNN5",       "Pearson_Smooth5"),
        ("S-KNN5",       "Spearman_Smooth5"),
    ]
    for i, (label, col) in enumerate(agg_cols):
        if col not in results_df.columns:
            continue
        vals = results_df[col].values
        order = np.argsort(vals)[::-1]
        parts = [f"{np.nanmean(vals[order[:b]]):>7.3f}" if b <= len(vals) else "    ---"
                 for b in fixed_buckets]
        parts.append(f"{np.nanmean(vals):>7.3f}")
        row = "  ".join(parts)
        prefix = f"  {'AGGREGATE':<18}" if i == 0 else f"  {'':18}"
        print(f"{prefix}  {label:<12}  {row}")

    print(SEP + "\n")


def compute_smoothed_for_k(spatial_df, targets_np, k):
    """
    Compute KNN-smoothed targets for a single k, one sample at a time.
    Returns (n_cells, n_genes) float32 array — one full copy of targets_np.
    Caller should del the result immediately after computing metrics.
    """
    smoothed = targets_np.copy()
    samples_arr = spatial_df["Sample"].values
    xs = spatial_df["Global_X"].values
    ys = spatial_df["Global_Y"].values

    for sample in tqdm(np.unique(samples_arr), desc=f"KNN-{k} smoothing", leave=False):
        mask = samples_arr == sample
        indices = np.where(mask)[0]
        if len(indices) < 2:
            continue
        coords = np.stack([xs[mask], ys[mask]], axis=1)
        tree = cKDTree(coords)
        actual_k = min(k + 1, len(indices))
        _, nn_local = tree.query(coords, k=actual_k)
        nn_global = indices[nn_local]          # (n, actual_k)
        smoothed[indices] = targets_np[nn_global].mean(axis=1)

    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Inference DeepSpot2Cell on Xenium/Single-Cell Data")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate (e.g., test, val, train)")
    args = parser.parse_args()

    global DEBUG_MODE
    DEBUG_MODE = args.debug
    print_system_status("Startup")

    cfg = load_config(Path(args.config).resolve())
    fix_seed(cfg["experiment"]["random_seed"])
    device = torch.device(args.device)

    data_cfg = cfg["data"]
    split_name = args.split # Grab the selected split
    
    eval_ids = data_cfg["split"][split_name]["ids"]

    check_data_exists(data_cfg["data_folder"], data_cfg["dataset_variant"], eval_ids)

    # --- 1. LOAD DATASET & MODEL ---
    print("\n" + "="*60)
    print("STEP 1: Running Model Inference")
    print("="*60)

    # Load Xenium panel gene indices first (fast, no dataset scan needed)
    print("\nLoading Xenium panel genes from stats CSV...")
    xenium_gene_indices, _ = load_xenium_gene_indices(data_cfg)
    gc.collect()

    # Init Dataset — pre-filtered to xenium genes only to avoid loading 18422 genes × all samples
    # Model output is still full 18422-gene space; preds are filtered after inference.
    test_ds = DS2CDataset(
        ids_list=eval_ids,
        cell_gt_available=data_cfg["split"][split_name].get("cell_gt_available", True),
        dataset_variant=data_cfg["dataset_variant"],
        data_path=data_cfg["data_folder"],
        model_name=data_cfg["model_name"],
        standard_scaling=cfg["normalization"]["standard_scaling"],
        normalize=cfg["normalization"]["normalize"],
        minmax=cfg["normalization"]["minmax"],
        norm_counts=cfg["normalization"]["norm_counts"],
        neighb_degree=cfg["model"].get("neighb_degree", 0),
        scellst=data_cfg.get("scellst", False),
        load_cell_types=data_cfg.get("load_cell_types", False),
        shuffle=False,
        cell_gene_indices=xenium_gene_indices.numpy(),
    )

    stats_path = Path(args.checkpoint).parent.parent / "dataset_stats.pt"
    if stats_path.exists():
        load_stats_from_file(stats_path, test_ds)

    print(f"\nLoading model from {args.checkpoint}...")
    model = DeepSpot2Cell.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, num_workers=4, shuffle=False)

    all_preds = []
    all_targets = []
    all_samples, all_xs, all_ys = [], [], []

    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            cell_emb = batch["cell_embeddings"].to(device).float()
            context = batch["neighb_embeddings"].to(device).float()
            context_mask = batch["neighb_masks"].to(device).bool()
            cell_true = batch["cell_expressions"].to(device).float()
            mask = batch["cell_mask"].to(device).float()
            cell_centroids = batch["cell_centroids"]

            barcodes = batch["patch_barcode"]

            s, x, y = parse_global_coordinates(barcodes, cell_centroids, mask.cpu().bool())
            all_samples.extend(s)
            all_xs.extend(x)
            all_ys.extend(y)

            for b in range(cell_emb.size(0)):
                # Ensure we only evaluate REAL cells, not padding
                valid_indices = mask[b] > 0
                if not valid_indices.any(): continue

                curr_cells = cell_emb[b, valid_indices]
                curr_ctx = context[b]
                curr_ctx_mask = context_mask[b]
                curr_targets = cell_true[b, valid_indices]

                preds = model._forward_single_cell(curr_cells, curr_ctx, curr_ctx_mask)

                # Filter preds to Xenium panel genes — model outputs full gene space (18422),
                # targets are already xenium-sized from the dataset (cell_gene_indices).
                if xenium_gene_indices is not None:
                    preds = preds[:, xenium_gene_indices]

                all_preds.append(preds.cpu())
                all_targets.append(curr_targets.cpu())

    if not all_preds:
        print("No valid cells found.")
        return

    print_system_status("Inference Done")

    # Cat preds and targets one at a time to halve peak memory at this step
    total_preds_raw = torch.cat(all_preds, dim=0)
    del all_preds
    gc.collect()
    total_targets_raw = torch.cat(all_targets, dim=0)
    del all_targets
    gc.collect()
    print_system_status("After Concat Cleanup")

    # Construct the dataframe exactly ONCE
    spatial_df = pd.DataFrame({
        "Sample": all_samples,
        "Global_X": np.concatenate(all_xs).astype(np.float32), 
        "Global_Y": np.concatenate(all_ys).astype(np.float32)
    })
    del all_samples, all_xs, all_ys

    print(f"Total Cells Evaluated: {total_preds_raw.shape[0]}")
    
    # --- INVERSE TRANSFORM ---
    print("Inverse transforming predictions in chunks...")
    chunk_size = 200_000 # Process 200k cells at a time
    
    for i in range(0, total_preds_raw.shape[0], chunk_size):
        end_idx = i + chunk_size
        
        # Process, cast back to float32 safely, and overwrite in-place
        total_preds_raw[i:end_idx] = torch.from_numpy(
            test_ds.inverse_transform(total_preds_raw[i:end_idx].numpy(), is_spot=False).astype(np.float32)
        )
        total_targets_raw[i:end_idx] = torch.from_numpy(
            test_ds.inverse_transform(total_targets_raw[i:end_idx].numpy(), is_spot=False).astype(np.float32)
        )

    print_system_status("After Inverse Transform")

    # Use in-place clamping (clamp_ instead of clamp)
    preds_log = total_preds_raw.clamp_(min=0.0)
    targets_log = total_targets_raw.clamp_(min=0.0)

    # --- GENE FILTERING ---
    gene_expression_sum = targets_log.sum(dim=0)
    valid_gene_indices = torch.where(gene_expression_sum > 0)[0]
    
    # Create the smaller, filtered contiguous tensors
    final_preds = preds_log[:, valid_gene_indices].contiguous()
    final_targets = targets_log[:, valid_gene_indices].contiguous()

    # CRITICAL: Delete references to the 4GB+ tensors before garbage collection
    del total_preds_raw
    del total_targets_raw
    del preds_log
    del targets_log

    gc.collect()
    print_system_status("After Gene Filter Cleanup")


    num_valid_genes = len(valid_gene_indices)

    # Prepare Names
    valid_gene_names = [f"Gene_{i}" for i in range(num_valid_genes)]
    if "ordered_genes_file" in data_cfg:
        gene_file = Path(data_cfg["data_folder"]) / data_cfg["ordered_genes_file"]
        if gene_file.exists():
            with open(gene_file, "r") as f:
                master_gene_list = json.load(f)
            # Apply standardization to master list
            master_gene_list = [clean_gene_string(g) for g in master_gene_list]
            # valid_gene_indices indexes into the (already-filtered) collected tensor.
            # If pre-scan filtered genes, map back through xenium_gene_indices to get
            # the correct offset into the full master list.
            if xenium_gene_indices is not None:
                master_offsets = xenium_gene_indices[valid_gene_indices].tolist()
            else:
                master_offsets = valid_gene_indices.tolist()
            valid_gene_names = [master_gene_list[i] for i in master_offsets]

    # --- DEBUG: PREDICTION SANITY CHECK (top-variance genes) ---
    print("\n" + "="*60)
    print("DEBUG: Prediction vs Ground Truth Sanity Check")
    gene_var = final_targets.var(dim=0).numpy()
    top_var_indices = np.argsort(gene_var)[::-1][:10]
    top_var_genes = [valid_gene_names[i] for i in top_var_indices]
    num_cells_total = final_preds.shape[0]
    random_cell_indices = np.random.choice(num_cells_total, min(10, num_cells_total), replace=False)
    print(f"Top-variance genes: {top_var_genes}")
    print(f"{'Gene':<18} {'mean_pred':>10} {'mean_gt':>10} {'std_pred':>10} {'std_gt':>10}")
    print("-" * 60)
    for g_idx in top_var_indices:
        p_vals = final_preds[random_cell_indices, g_idx].numpy()
        t_vals = final_targets[random_cell_indices, g_idx].numpy()
        print(f"{valid_gene_names[g_idx]:<18} {p_vals.mean():>10.4f} {t_vals.mean():>10.4f} "
              f"{p_vals.std():>10.4f} {t_vals.std():>10.4f}")
    print("="*60 + "\n")

    # --- METRICS ---
    # All gene-level and cell-level metrics computed in chunks to avoid
    # OOM from argsort int64 intermediates on large (n_cells, n_genes) matrices.
    print("Computing per-gene and per-cell metrics (chunked)...")

    # Use numpy views of torch tensors — no extra allocation for the base arrays
    preds_np   = final_preds.numpy()    # float32, shares storage with tensor
    targets_np = final_targets.numpy()  # float32, shares storage with tensor
    gene_variances = targets_np.var(axis=0)

    GENE_CHUNK = 500   # processes 500 genes × n_cells at a time
    CELL_CHUNK = 10_000

    n_total_cells = preds_np.shape[0]

    per_gene_pearson          = np.zeros(num_valid_genes, dtype=np.float32)
    per_gene_spearman         = np.zeros(num_valid_genes, dtype=np.float32)
    per_gene_spearman_clipped = np.zeros(num_valid_genes, dtype=np.float32)

    # preds_clipped: one full copy — same size as preds_np but no argsort intermediates yet
    preds_clipped = np.where(preds_np < CLIP_THRESHOLD, 0.0, preds_np).astype(np.float32)

    for g0 in tqdm(range(0, num_valid_genes, GENE_CHUNK), desc="Per-gene metrics"):
        g1 = min(g0 + GENE_CHUNK, num_valid_genes)
        fp_c  = preds_np[:, g0:g1]
        ft_c  = targets_np[:, g0:g1]
        fpc_c = preds_clipped[:, g0:g1]
        per_gene_pearson[g0:g1]          = _pearson_per_gene(fp_c, ft_c)
        per_gene_spearman[g0:g1]         = _spearman_per_gene(fp_c, ft_c)
        per_gene_spearman_clipped[g0:g1] = _spearman_per_gene(fpc_c, ft_c)
    del preds_clipped

    # Per-cell: correlation across genes for each cell (chunk over cells)
    cell_pearson  = np.zeros(n_total_cells, dtype=np.float32)
    cell_spearman = np.zeros(n_total_cells, dtype=np.float32)
    for c0 in tqdm(range(0, n_total_cells, CELL_CHUNK), desc="Per-cell metrics"):
        c1 = min(c0 + CELL_CHUNK, n_total_cells)
        # Transpose slice: (n_genes, cell_chunk) — argsort along n_genes axis = small int64
        cell_pearson[c0:c1]  = _pearson_per_gene(preds_np[c0:c1].T, targets_np[c0:c1].T)
        cell_spearman[c0:c1] = _spearman_per_gene(preds_np[c0:c1].T, targets_np[c0:c1].T)

    print(f"\n  Per-cell Pearson:  mean={np.nanmean(cell_pearson):.4f}  "
          f"med={np.nanmedian(cell_pearson):.4f}  std={np.nanstd(cell_pearson):.4f}  "
          f"n={n_total_cells:,}")
    print(f"  Per-cell Spearman: mean={np.nanmean(cell_spearman):.4f}  "
          f"med={np.nanmedian(cell_spearman):.4f}  std={np.nanstd(cell_spearman):.4f}\n")
    del cell_pearson, cell_spearman

    print_system_status("After All Metrics Calc")

    # --- SPATIAL NEIGHBOR SMOOTHING (k=3 then k=5 sequentially to halve peak RAM) ---
    print("\nComputing Spatial Neighbor Smoothed Metrics (k=3, k=5)...")
    neighbor_metrics = {}
    for k in (3, 5):
        smooth_t = compute_smoothed_for_k(spatial_df, targets_np, k)
        p_s = np.zeros(num_valid_genes, dtype=np.float32)
        s_s = np.zeros(num_valid_genes, dtype=np.float32)
        for g0 in range(0, num_valid_genes, GENE_CHUNK):
            g1 = min(g0 + GENE_CHUNK, num_valid_genes)
            p_s[g0:g1] = _pearson_per_gene(preds_np[:, g0:g1], smooth_t[:, g0:g1])
            s_s[g0:g1] = _spearman_per_gene(preds_np[:, g0:g1], smooth_t[:, g0:g1])
        neighbor_metrics[k] = (p_s, s_s)
        del smooth_t
        gc.collect()

    del preds_np, targets_np
    gc.collect()
    print_system_status("After Smoothed Metrics Calc")

    # --- DATA MERGING ---
    results_df = pd.DataFrame({
        "Gene": valid_gene_names,
        "Pearson": per_gene_pearson,
        "Spearman": per_gene_spearman,
        "Spearman_Clipped": per_gene_spearman_clipped,
        "Pearson_Smooth3": neighbor_metrics[3][0],
        "Spearman_Smooth3": neighbor_metrics[3][1],
        "Pearson_Smooth5": neighbor_metrics[5][0],
        "Spearman_Smooth5": neighbor_metrics[5][1],
        "Test_Set_Variance": gene_variances,
    })

    # Save
    output_dir = Path(args.checkpoint).parent / f"Xenium_{split_name}_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "xenium_full_metrics.csv", index=False)

    buckets = [10, 50, 100, 200, 300, 500, num_valid_genes]
    buckets = [b for b in buckets if b <= num_valid_genes]

    # --- PER-SAMPLE SUMMARY ---
    print_per_sample_summary(final_preds, final_targets, spatial_df, results_df, valid_gene_names)

    # --- PLOTTING (ceiling only) ---
    for metric_col, color, label in [
        ("Pearson",           "purple", "Mean Pearson"),
        ("Spearman",          "purple", "Mean Spearman"),
        ("Spearman_Clipped",  "navy",   "Mean Spearman (clipped)"),
        ("Pearson_Smooth3",   "green",  "Mean Pearson KNN-3"),
        ("Spearman_Smooth3",  "green",  "Mean Spearman KNN-3"),
        ("Pearson_Smooth5",   "red",    "Mean Pearson KNN-5"),
        ("Spearman_Smooth5",  "red",    "Mean Spearman KNN-5"),
    ]:
        df_sorted = results_df.sort_values(by=metric_col, ascending=False)
        safe = metric_col.lower()
        create_staggered_plot(
            df_sorted, buckets,
            f"Xenium Ceiling ({metric_col})",
            f"ceiling_{safe}.png",
            output_dir, color=color, label=label, metric_col=metric_col,
        )

    # --- SPATIAL PLOTS ---
    # Top 4 genes by Pearson among top-50 by variance
    df_bio = results_df.sort_values(by="Test_Set_Variance", ascending=False).head(50)
    top_spatial = df_bio.sort_values(by="Pearson", ascending=False).head(4)
    
    top_genes_names = top_spatial["Gene"].tolist()
    top_indices = [valid_gene_names.index(g) for g in top_genes_names]

    if top_indices:
        # Slice the tensors FIRST, then convert to numpy (Saves ~10GB RAM)
        subset_preds = final_preds[:, top_indices].numpy()
        subset_targets = final_targets[:, top_indices].numpy()
        
        # Because we sliced the array, the new indices for the plotting function are just 0, 1, 2, 3...
        new_indices = list(range(len(top_indices)))
        
        plot_spatial_genes_xenium(
            spatial_df, 
            subset_preds, 
            subset_targets, 
            top_genes_names, 
            new_indices, 
            output_dir
        )
    
    print(f"\nDone. Results saved to {output_dir}")

if __name__ == "__main__":
    main()