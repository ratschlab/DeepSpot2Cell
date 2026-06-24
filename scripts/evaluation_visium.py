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
from collections import defaultdict
import gc
import lightning as L
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ConstantInputWarning
import warnings
import scanpy as sc
import anndata as ad
from hest import iter_hest

from deepspot2cell import DeepSpot2Cell, DS2CDataset
from deepspot2cell.utils.utils import (
    fix_seed, 
    standardize_gene_names,
    clean_gene_string, 
    is_artifact
)


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
    stats = torch.load(stats_path)
    
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
    else:
        test_ds.minmax = False


def get_test_set_raw_hvg_votes(data_folder, sample_ids):
    print("\n" + "="*60)
    print(f"STEP 1: Calculating Biological Consensus on RAW Data ({len(sample_ids)} samples)")
    print("="*60)
    
    gene_hvg_votes = defaultdict(int)
    gene_presence_count = defaultdict(int)
    
    # Iterate exactly like in generate_hybrid_gene_list
    for i, sample_id in enumerate(sample_ids):
        iterator = iter_hest(data_folder, id_list=[sample_id], load_transcripts=False)
        
        try:
            st = next(iterator)
            adata = st.adata
            
            # 1. Standardize Names
            adata = standardize_gene_names(adata)

            # 2. Filter (Same as training)
            sc.pp.filter_genes(adata, min_cells=10) 
            current_sample_genes = adata.var_names.tolist()
            
            for gene in current_sample_genes:
                gene_presence_count[gene] += 1

            # 3. HVG Selection (On RAW counts)
            # Ensure n_top_genes doesn't exceed available genes
            n_target = min(5000, adata.n_vars)
            
            sc.pp.highly_variable_genes(
                adata, 
                n_top_genes=n_target, 
                flavor='seurat_v3',
                span=0.3
            )
            
            hvg_mask = adata.var['highly_variable']
            hvg_genes = adata.var_names[hvg_mask].tolist()

            for gene in hvg_genes:
                gene_hvg_votes[gene] += 1
            
            print(f"  [{i+1}/{len(sample_ids)}] {sample_id}: Found {len(hvg_genes)} HVGs")
            
            del st, adata
            gc.collect()

        except Exception as e:
            print(f"   ⚠️ Skipped {sample_id}: {e}")
            continue

    # Convert to DataFrame
    vote_series = pd.Series(gene_hvg_votes, name='test_hvg_votes')
    presence_series = pd.Series(gene_presence_count, name='test_presence_count')

    df_res = pd.concat([vote_series, presence_series], axis=1).fillna(0).astype(int)
    
    # Clean index
    df_res.index = [clean_gene_string(g) for g in df_res.index]
    
    # Handle duplicates if clean_gene_string merged any (sum votes)
    df_res = df_res.groupby(df_res.index).sum()
    
    print(f"Consensus calculated. Max votes: {df_res['test_hvg_votes'].max()}\n")
    return df_res


def parse_patch_barcodes(barcodes):
    parsed_data = []
    for bc in barcodes:
        parts = bc.split('_')
        row = int(parts[1])
        col = int(parts[2])
        sample_id = "_".join(parts[3:])
        parsed_data.append({"Sample": sample_id, "Row": row, "Col": col})
    return pd.DataFrame(parsed_data)


def bootstrap_se(values, n_bootstraps=1000):
    if len(values) < 2: return 0.0
    values = np.array(values)
    boot_means = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    return np.std(boot_means)


def plot_spatial_genes(spatial_df, preds, targets, gene_names, top_genes_indices, output_dir):
    samples = spatial_df["Sample"].unique()
    samples_to_plot = samples[:8]
    
    for sample in samples_to_plot:
        sample_mask = spatial_df["Sample"] == sample
        sample_rows = spatial_df.loc[sample_mask, "Row"].values
        sample_cols = spatial_df.loc[sample_mask, "Col"].values
        
        sample_preds = preds[sample_mask]
        sample_targets = targets[sample_mask]
        
        num_genes_to_plot = len(top_genes_indices)
        fig, axes = plt.subplots(num_genes_to_plot, 2, figsize=(12, 5 * num_genes_to_plot))
        fig.suptitle(f"Spatial Prediction vs Ground Truth: {sample}", fontsize=16)
        
        if num_genes_to_plot == 1: axes = np.array([axes])

        for i in range(num_genes_to_plot):
            gene_idx = top_genes_indices[i]
            gene_name = gene_names[gene_idx]
            
            p_vals = sample_preds[:, gene_idx]
            t_vals = sample_targets[:, gene_idx]
            
            r_val = 0.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConstantInputWarning)
                try:
                    stat, _ = pearsonr(p_vals.flatten(), t_vals.flatten())
                    if not np.isnan(stat): r_val = stat
                except: r_val = 0.0

            vmin = min(np.percentile(p_vals, 1), np.percentile(t_vals, 1))
            vmax = max(np.percentile(p_vals, 99), np.percentile(t_vals, 99))
            
            sc1 = axes[i, 0].scatter(sample_cols, -sample_rows, c=p_vals, cmap="magma", s=15, vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f"Pred: {gene_name} (Sample R={r_val:.2f})")
            axes[i, 0].axis('off')
            plt.colorbar(sc1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            sc2 = axes[i, 1].scatter(sample_cols, -sample_rows, c=t_vals, cmap="magma", s=15, vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f"GT: {gene_name}")
            axes[i, 1].axis('off')
            plt.colorbar(sc2, ax=axes[i, 1], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = output_dir / f"spatial_plot_{sample}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()


def create_staggered_plot(sorted_df, buckets, title, filename, output_dir, color='blue', label='Mean Pearson', metric_col="Pearson", ylabel="Pearson R"):
    means, ses = [], []
    valid_buckets = [b for b in buckets if b <= len(sorted_df)]
    for k in valid_buckets:
        vals = sorted_df.head(k)[metric_col].values # Dynamic metric column
        means.append(np.mean(vals))
        ses.append(bootstrap_se(vals))
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(range(len(valid_buckets)), means, yerr=ses, fmt='-o', capsize=5, color=color, label=label)
    plt.xticks(range(len(valid_buckets)), valid_buckets)
    plt.title(title)
    plt.xlabel("Top K Genes")
    plt.ylabel(ylabel) # Dynamic Y Label
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / filename)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inference DeepSpot2Cell on VISIUM Data")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    fix_seed(cfg["experiment"]["random_seed"])
    device = torch.device(args.device)
    data_cfg = cfg["data"]
    
    if args.split not in data_cfg["split"]:
        print(f"Error: Split '{args.split}' not defined in config.")
        return
    
    target_ids = data_cfg["split"][args.split]["ids"]
    check_data_exists(data_cfg["data_folder"], data_cfg["dataset_variant"], target_ids)

    # --- 1. GET RAW HVG STATS (INDEPENDENT OF MODEL) ---
    hvg_df = get_test_set_raw_hvg_votes(data_cfg["data_folder"], target_ids)

    # --- 2. LOAD DATASET & MODEL (FOR INFERENCE) ---
    print("\n" + "="*60)
    print("STEP 2: Running Model Inference")
    print("="*60)
    
    test_ds = DS2CDataset(
        ids_list=target_ids,
        cell_gt_available=False, 
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
    )

    stats_path = Path(args.checkpoint).parent.parent / "dataset_stats.pt"
    if stats_path.exists():
        load_stats_from_file(stats_path, test_ds)
    
    model = DeepSpot2Cell.load_from_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, num_workers=4, shuffle=False)

    all_preds = []
    all_targets = []
    all_barcodes = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            cell_emb = batch["cell_embeddings"].to(device).float()
            mask = batch["cell_mask"].to(device).float()
            context = batch["neighb_embeddings"].to(device).float()
            context_mask = batch["neighb_masks"].to(device).bool()
            spot_true = batch["spot_expression"].to(device).float()
            is_inside = batch["is_inside_spot"].to(device).float()
            
            cell_emb_masked = cell_emb * is_inside.unsqueeze(-1)
            mask_inspot = mask * is_inside

            preds = model(cell_emb_masked, mask_inspot, context, context_mask)
            
            all_preds.append(preds.cpu())
            all_targets.append(spot_true.cpu())
            all_barcodes.extend(batch["patch_barcode"])

    if not all_preds:
        print("No valid data found.")
        return

    total_preds_raw = torch.cat(all_preds, dim=0)
    total_targets_raw = torch.cat(all_targets, dim=0)

    # Inverse Transform
    print("Inverse transforming predictions...")
    preds_log = torch.from_numpy(test_ds.inverse_transform(total_preds_raw.numpy(), is_spot=True))
    targets_log = torch.from_numpy(test_ds.inverse_transform(total_targets_raw.numpy(), is_spot=True))
    
    # Ensure non-negative
    preds_log = preds_log.clamp(min=0.0)
    targets_log = targets_log.clamp(min=0.0)

    # Filter Valid Genes
    gene_expression_sum = targets_log.sum(dim=0)
    valid_gene_indices = torch.where(gene_expression_sum > 0)[0]
    
    final_preds = preds_log[:, valid_gene_indices]
    final_targets = targets_log[:, valid_gene_indices]
    num_valid_genes = len(valid_gene_indices)

    # Get Gene Names
    valid_gene_names = [f"Gene_{i}" for i in range(num_valid_genes)]
    if "ordered_genes_file" in data_cfg:
        gene_file = Path(data_cfg["data_folder"]) / data_cfg["ordered_genes_file"]
        if gene_file.exists():
            with open(gene_file, "r") as f:
                master_gene_list = json.load(f)
            # Standardize names
            master_gene_list = [clean_gene_string(g) for g in master_gene_list]
            valid_gene_names = [master_gene_list[i] for i in valid_gene_indices.tolist()]

    # --- DEBUG: PREDICTION SANITY CHECK ---
    print("\n" + "="*60)
    print("DEBUG: Prediction vs Ground Truth Sanity Check (Top 10 HVGs)")
    
    # 1. Identify Top 10 HVGs from Consensus
    hvg_sorted = hvg_df.sort_values(by="test_hvg_votes", ascending=False)
    valid_hvg_names = [g for g in hvg_sorted.index if g in valid_gene_names]
    top_10_hvgs = valid_hvg_names[:30] # Already set to 10
    
    if not top_10_hvgs:
        print("Warning: No overlap between Consensus HVGs and Valid Genes for Debug Print.")
    else:
        # 2. Pick 30 Random Spots
        num_spots_total = final_preds.shape[0]
        n_samples = min(30, num_spots_total) 
        random_spot_indices = np.random.choice(num_spots_total, n_samples, replace=False)
        
        print(f"Showing {n_samples} random spots for Top {len(top_10_hvgs)} HVGs:")
        print(f"Top HVGs: {top_10_hvgs}")
        
        for gene_name in top_10_hvgs:
            g_idx = valid_gene_names.index(gene_name)
            
            p_vals = final_preds[random_spot_indices, g_idx].numpy()
            t_vals = final_targets[random_spot_indices, g_idx].numpy()
            
            print(f"\nGene: {gene_name}")
            print(f"{'Spot Idx':<10} | {'Pred':<10} | {'True':<10} | {'Diff':<10}")
            print("-" * 46)
            for s_idx, p, t in zip(random_spot_indices, p_vals, t_vals):
                print(f"{s_idx:<10} | {p:<10.4f} | {t:<10.4f} | {p-t:<10.4f}")
    
    print("="*60 + "\n")
    # --------------------------------------

    # Metric Calculation
    # 1. Pearson
    print("Calculating Pearson Correlations...")
    metric_p = PearsonCorrCoef(num_outputs=final_preds.shape[1]).to(final_preds.device)
    per_gene_pearson = metric_p(final_preds, final_targets)
    per_gene_pearson = torch.nan_to_num(per_gene_pearson, nan=0.0).cpu().numpy()
    
    # Cleanup Pearson to free memory
    del metric_p
    gc.collect()

    # 2. Spearman
    print("Calculating Spearman Correlations...")
    metric_s = SpearmanCorrCoef(num_outputs=final_preds.shape[1]).to(final_preds.device)
    per_gene_spearman = metric_s(final_preds, final_targets)
    per_gene_spearman = torch.nan_to_num(per_gene_spearman, nan=0.0).cpu().numpy()

    # Cleanup Spearman
    del metric_s
    gc.collect()
    
    test_gene_variances = final_targets.var(dim=0).cpu().numpy()

    # --- 3. MERGE METRICS WITH RAW HVG STATS ---
    results_df = pd.DataFrame({
        "Gene": valid_gene_names,
        "Pearson": per_gene_pearson,
        "Spearman": per_gene_spearman, # Added Spearman
        "Test_Set_Variance": test_gene_variances
    })
    
    # Merge Test Set HVG info (Left join to keep only genes present in model)
    results_df = results_df.merge(hvg_df, left_on="Gene", right_index=True, how="left")
    results_df['test_hvg_votes'] = results_df['test_hvg_votes'].fillna(0)
    
    # Merge Training Votes
    votes_file_path = None
    if "ordered_genes_file" in data_cfg:
        gene_filename = data_cfg["ordered_genes_file"]
        # Assumes format like "hybrid_genes.json" -> "hybrid_genes_votes.csv"
        votes_filename = Path(gene_filename).stem + "_stats.csv"
        
        potential_path = Path(data_cfg["data_folder"]) / votes_filename
        if potential_path.exists():
            votes_file_path = potential_path

    if votes_file_path:
        print(f"Loading Training HVG Votes from {votes_file_path}...")
        votes_df = pd.read_csv(votes_file_path, index_col=0)
        votes_df.index = [clean_gene_string(g) for g in votes_df.index]
        
        results_df = results_df.merge(votes_df, left_on="Gene", right_index=True, how="left")
        
        results_df['hvg_votes'] = results_df['hvg_votes'].fillna(0)
        if 'in_xenium' not in results_df.columns and 'in_xenium' in votes_df.columns:
             results_df['in_xenium'] = votes_df['in_xenium']
        results_df['in_xenium'] = results_df['in_xenium'].fillna(False).astype(bool)
    else:
        print("No training votes file found (e.g., '_votes.csv'). Skipping training stats.")
        results_df['hvg_votes'] = 0
        results_df['in_xenium'] = False

    output_dir = Path(args.checkpoint).parent / f"Visium_{args.split}_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / f"visium_{args.split}_full_metrics.csv", index=False)

    buckets = [10, 50, 100, 300, 500, 1000, 2000, num_valid_genes]
    buckets = [b for b in buckets if b <= num_valid_genes]

    # ==========================================
    #           PEARSON PLOTS
    # ==========================================
    
    # Plot A: Biological Relevance (Sorted by RAW TEST HVG VOTES)
    df_sorted_hvg = results_df.sort_values(by=["test_hvg_votes", "Test_Set_Variance"], ascending=[False, False])
    create_staggered_plot(df_sorted_hvg, buckets, 
                          f"Plot A: Biological Relevance (Pearson)\n(Sorted by {args.split} Consensus HVG Votes)",
                          "plot_A_biological_relevance_pearson.png", output_dir, metric_col="Pearson", ylabel="Pearson R")

    # Plot B: Generalization (Sorted by TRAIN VOTES)
    if results_df['hvg_votes'].sum() > 0:
        df_sorted_votes = results_df.sort_values(by="hvg_votes", ascending=False)
        create_staggered_plot(df_sorted_votes, buckets,
                              "Plot B: Generalization (Pearson)\n(Sorted by Training HVG Consensus)",
                              "plot_B_generalization_pearson.png", output_dir, color='orange', metric_col="Pearson", ylabel="Pearson R")

    # Plot C: Utility (Xenium vs Visium, Sorted by Test HVG)
    if results_df['in_xenium'].sum() > 0:
        xenium_df_hvg = results_df[results_df['in_xenium']].sort_values(by=["test_hvg_votes", "Test_Set_Variance"], ascending=[False, False])
        visium_df_hvg = results_df[~results_df['in_xenium']].sort_values(by=["test_hvg_votes", "Test_Set_Variance"], ascending=[False, False])
        
        plt.figure(figsize=(10, 6))
        
        valid_buckets_x = [b for b in buckets if b <= len(xenium_df_hvg)]
        x_means = [np.mean(xenium_df_hvg.head(k)["Pearson"].values) for k in valid_buckets_x]
        x_ses = [bootstrap_se(xenium_df_hvg.head(k)["Pearson"].values) for k in valid_buckets_x]
        plt.errorbar(range(len(valid_buckets_x)), x_means, yerr=x_ses, fmt='-o', color='green', capsize=5, label='Xenium Genes (sorted by Bio Relevance)')

        valid_buckets_v = [b for b in buckets if b <= len(visium_df_hvg)]
        v_means = [np.mean(visium_df_hvg.head(k)["Pearson"].values) for k in valid_buckets_v]
        v_ses = [bootstrap_se(visium_df_hvg.head(k)["Pearson"].values) for k in valid_buckets_v]
        plt.errorbar(range(len(valid_buckets_v)), v_means, yerr=v_ses, fmt='-x', color='gray', capsize=5, label='Visium Fillers (sorted by Bio Relevance)')

        plt.xticks(range(len(buckets)), buckets)
        plt.title(f"Plot C: Utility on {args.split} Set (Pearson)")
        plt.xlabel("Top K Genes")
        plt.ylabel("Pearson R")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "plot_C_utility_comparison_pearson.png")
        plt.close()

    # Plot D: Overall Model Ceiling
    df_sorted_r = results_df.sort_values(by="Pearson", ascending=False)
    create_staggered_plot(df_sorted_r, buckets,
                          "Plot D: Overall Model Ceiling (Pearson)\n(Best Predicted Genes)",
                          "plot_D_best_performance_pearson.png", output_dir, color='purple', metric_col="Pearson", ylabel="Pearson R")

    # Plot E: Ceiling Breakdown
    if results_df['in_xenium'].sum() > 0:
        xenium_df_best = results_df[results_df['in_xenium']].sort_values(by="Pearson", ascending=False)
        visium_df_best = results_df[~results_df['in_xenium']].sort_values(by="Pearson", ascending=False)
        
        plt.figure(figsize=(10, 6))
        
        valid_buckets_all = [b for b in buckets if b <= len(df_sorted_r)]
        all_means = [np.mean(df_sorted_r.head(k)["Pearson"].values) for k in valid_buckets_all]
        plt.plot(range(len(valid_buckets_all)), all_means, '--', color='purple', alpha=0.5, label='Overall Ceiling')

        valid_buckets_x = [b for b in buckets if b <= len(xenium_df_best)]
        x_means = [np.mean(xenium_df_best.head(k)["Pearson"].values) for k in valid_buckets_x]
        x_ses = [bootstrap_se(xenium_df_best.head(k)["Pearson"].values) for k in valid_buckets_x]
        plt.errorbar(range(len(valid_buckets_x)), x_means, yerr=x_ses, fmt='-o', color='green', capsize=5, label='Best Predicted Xenium')

        valid_buckets_v = [b for b in buckets if b <= len(visium_df_best)]
        v_means = [np.mean(visium_df_best.head(k)["Pearson"].values) for k in valid_buckets_v]
        v_ses = [bootstrap_se(visium_df_best.head(k)["Pearson"].values) for k in valid_buckets_v]
        plt.errorbar(range(len(valid_buckets_v)), v_means, yerr=v_ses, fmt='-x', color='gray', capsize=5, label='Best Predicted Visium')

        plt.xticks(range(len(buckets)), buckets)
        plt.title("Plot E: Panel Ceiling Breakdown (Pearson)")
        plt.xlabel("Top K Genes")
        plt.ylabel("Pearson R")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "plot_E_ceiling_breakdown_pearson.png")
        plt.close()


    # ==========================================
    #           SPEARMAN PLOTS
    # ==========================================

    # Plot A: Biological Relevance (Sorted by RAW TEST HVG VOTES) - using Spearman metric
    create_staggered_plot(df_sorted_hvg, buckets, 
                          f"Plot A: Biological Relevance (Spearman)\n(Sorted by {args.split} Consensus HVG Votes)",
                          "plot_A_biological_relevance_spearman.png", output_dir, metric_col="Spearman", ylabel="Spearman R")

    # Plot B: Generalization (Sorted by TRAIN VOTES) - using Spearman metric
    if results_df['hvg_votes'].sum() > 0:
        create_staggered_plot(df_sorted_votes, buckets,
                              "Plot B: Generalization (Spearman)\n(Sorted by Training HVG Consensus)",
                              "plot_B_generalization_spearman.png", output_dir, color='orange', metric_col="Spearman", ylabel="Spearman R")

    # Plot C: Utility (Xenium vs Visium, Sorted by Test HVG) - using Spearman metric
    if results_df['in_xenium'].sum() > 0:
        # Use existing sorted DataFrames df_sorted_hvg subsetted
        # xenium_df_hvg and visium_df_hvg are already sorted by Votes from Pearson section
        
        plt.figure(figsize=(10, 6))
        
        valid_buckets_x = [b for b in buckets if b <= len(xenium_df_hvg)]
        # Change column to "Spearman"
        x_means = [np.mean(xenium_df_hvg.head(k)["Spearman"].values) for k in valid_buckets_x]
        x_ses = [bootstrap_se(xenium_df_hvg.head(k)["Spearman"].values) for k in valid_buckets_x]
        plt.errorbar(range(len(valid_buckets_x)), x_means, yerr=x_ses, fmt='-o', color='green', capsize=5, label='Xenium Genes (sorted by Bio Relevance)')

        valid_buckets_v = [b for b in buckets if b <= len(visium_df_hvg)]
        v_means = [np.mean(visium_df_hvg.head(k)["Spearman"].values) for k in valid_buckets_v]
        v_ses = [bootstrap_se(visium_df_hvg.head(k)["Spearman"].values) for k in valid_buckets_v]
        plt.errorbar(range(len(valid_buckets_v)), v_means, yerr=v_ses, fmt='-x', color='gray', capsize=5, label='Visium Fillers (sorted by Bio Relevance)')

        plt.xticks(range(len(buckets)), buckets)
        plt.title(f"Plot C: Utility on {args.split} Set (Spearman)")
        plt.xlabel("Top K Genes")
        plt.ylabel("Spearman R")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "plot_C_utility_comparison_spearman.png")
        plt.close()

    # Plot D: Overall Model Ceiling - Re-sort by Spearman!
    df_sorted_s = results_df.sort_values(by="Spearman", ascending=False)
    create_staggered_plot(df_sorted_s, buckets,
                          "Plot D: Overall Model Ceiling (Spearman)\n(Best Predicted Genes)",
                          "plot_D_best_performance_spearman.png", output_dir, color='purple', metric_col="Spearman", ylabel="Spearman R")

    # Plot E: Ceiling Breakdown - Re-sort subsets by Spearman!
    if results_df['in_xenium'].sum() > 0:
        xenium_df_best_s = results_df[results_df['in_xenium']].sort_values(by="Spearman", ascending=False)
        visium_df_best_s = results_df[~results_df['in_xenium']].sort_values(by="Spearman", ascending=False)
        
        plt.figure(figsize=(10, 6))
        
        valid_buckets_all = [b for b in buckets if b <= len(df_sorted_s)]
        all_means = [np.mean(df_sorted_s.head(k)["Spearman"].values) for k in valid_buckets_all]
        plt.plot(range(len(valid_buckets_all)), all_means, '--', color='purple', alpha=0.5, label='Overall Ceiling')

        valid_buckets_x = [b for b in buckets if b <= len(xenium_df_best_s)]
        x_means = [np.mean(xenium_df_best_s.head(k)["Spearman"].values) for k in valid_buckets_x]
        x_ses = [bootstrap_se(xenium_df_best_s.head(k)["Spearman"].values) for k in valid_buckets_x]
        plt.errorbar(range(len(valid_buckets_x)), x_means, yerr=x_ses, fmt='-o', color='green', capsize=5, label='Best Predicted Xenium')

        valid_buckets_v = [b for b in buckets if b <= len(visium_df_best_s)]
        v_means = [np.mean(visium_df_best_s.head(k)["Spearman"].values) for k in valid_buckets_v]
        v_ses = [bootstrap_se(visium_df_best_s.head(k)["Spearman"].values) for k in valid_buckets_v]
        plt.errorbar(range(len(valid_buckets_v)), v_means, yerr=v_ses, fmt='-x', color='gray', capsize=5, label='Best Predicted Visium')

        plt.xticks(range(len(buckets)), buckets)
        plt.title("Plot E: Panel Ceiling Breakdown (Spearman)")
        plt.xlabel("Top K Genes")
        plt.ylabel("Spearman R")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "plot_E_ceiling_breakdown_spearman.png")
        plt.close()

    # --- SPATIAL PLOTS ---
    # Top 4 genes that are BIOLOGICALLY RELEVANT (Top 500 votes) AND WELL PREDICTED
    df_bio = results_df.sort_values(by=["test_hvg_votes", "Test_Set_Variance"], ascending=[False, False]).head(500)
    top_spatial = df_bio.sort_values(by="Pearson", ascending=False).head(4)
    
    top_genes_names = top_spatial["Gene"].tolist()
    top_indices = [valid_gene_names.index(g) for g in top_genes_names]

    if top_indices:
        spatial_df = parse_patch_barcodes(all_barcodes)
        plot_spatial_genes(
            spatial_df, 
            final_preds.cpu().numpy(), 
            final_targets.cpu().numpy(), 
            valid_gene_names, 
            top_indices, 
            output_dir
        )

    print(f"\nDone. Results saved to {output_dir}")

if __name__ == "__main__":
    main()