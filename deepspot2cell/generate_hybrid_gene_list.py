import argparse
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from hest import iter_hest
from pathlib import Path
import yaml
import gc
import warnings
from scipy import sparse
from collections import defaultdict

from deepspot2cell.utils.utils import (
    load_config, 
    standardize_gene_names, 
    clean_gene_string, 
    is_artifact
)


def generate_hybrid_gene_list(
    config_path,
    target_count=5000,
    organ="bowel",
    vis_presence_thresh=0.50,
    vis_sparsity_thresh=0.60,
    min_xenium_cells_per_gene=100,
    min_xenium_samples_per_gene=1,
):
    print(f"Generating Hybrid Gene List (Target: {target_count})")
    cfg = load_config(config_path)
    data_folder = cfg["data"]["data_folder"]
    
    splits = cfg.get("data", {}).get("split", {})
    train_ids = splits.get("train", {}).get("ids", [])
    
    test_split = splits.get("test", {})
    test_ids = test_split.get("ids", []) if test_split.get("type") == "xenium" else []

    qv_thr = cfg.get('processing', {}).get('qv_thr', 20)

    # =========================================================================
    # STEP 1: Collect Xenium genes per sample (union — no intersection)
    # =========================================================================
    xenium_genes = set()
    per_sample_xenium = {}  # sample_id -> set of genes

    if not test_ids:
        print("⚠️ No Xenium test samples found. Proceeding with Visium-only HVG selection.")
    else:
        print(f"Extracting Xenium panels from {len(test_ids)} test samples...")

        for tid in test_ids:
            try:
                for st in iter_hest(data_folder, id_list=[tid], load_transcripts=True):
                    df = st.transcript_df
                    if 'qv' in df.columns:
                        df['qv'] = pd.to_numeric(df['qv'], errors='coerce')
                        df = df[df['qv'] > qv_thr]

                    # Keep only cell-assigned transcripts
                    df = df[~df['cell_id'].isin(['UNASSIGNED', -1]) & df['cell_id'].notna()].copy()

                    # Clean gene names and remove artifact probes
                    df['feature_name'] = df['feature_name'].apply(clean_gene_string)
                    df = df[~df['feature_name'].apply(is_artifact)]

                    n_cells_this_sample = df['cell_id'].nunique()

                    # Count how many distinct cells express each gene (at least 1 transcript)
                    gene_cell_counts = df.groupby('feature_name')['cell_id'].nunique()

                    # Filter: gene must be expressed in at least min_xenium_cells_per_gene cells
                    passing_genes = set(
                        gene_cell_counts[gene_cell_counts >= min_xenium_cells_per_gene].index.tolist()
                    )

                    n_total_genes = len(gene_cell_counts)
                    n_dropped = n_total_genes - len(passing_genes)
                    print(f"  {tid}: {n_cells_this_sample:,} cells, {n_total_genes} panel genes → "
                          f"{len(passing_genes)} pass min_cells>={min_xenium_cells_per_gene} "
                          f"(dropped {n_dropped} sparse genes)")

                    per_sample_xenium[tid] = passing_genes
                    xenium_genes.update(passing_genes)

            except Exception as e:
                print(f"  ⚠️ Skipped Xenium sample {tid}: {e}")

        print(f"\nXenium union (after per-sample sparsity filter): "
              f"{len(xenium_genes)} unique genes across {len(per_sample_xenium)} samples")

        # Log pairwise panel overlaps for transparency
        if len(per_sample_xenium) > 1:
            sample_ids = list(per_sample_xenium.keys())
            for i, s1 in enumerate(sample_ids):
                for s2 in sample_ids[i+1:]:
                    overlap = len(per_sample_xenium[s1] & per_sample_xenium[s2])
                    print(f"  Overlap {s1} ∩ {s2}: {overlap} genes")

        # Cross-sample filter: gene must pass sparsity threshold in >= min_xenium_samples_per_gene samples
        if min_xenium_samples_per_gene > 1 and len(per_sample_xenium) >= min_xenium_samples_per_gene:
            gene_sample_pass_counts = defaultdict(int)
            for sid, genes in per_sample_xenium.items():
                for g in genes:
                    gene_sample_pass_counts[g] += 1

            before_cross = len(xenium_genes)
            xenium_genes = {
                g for g, cnt in gene_sample_pass_counts.items()
                if cnt >= min_xenium_samples_per_gene
            }
            print(f"  Cross-sample filter (>={min_xenium_samples_per_gene} samples): "
                  f"{before_cross} → {len(xenium_genes)} genes "
                  f"(dropped {before_cross - len(xenium_genes)})")

    # =========================================================================
    # STEP 2: Scan Visium training samples — track gene presence & HVG votes
    # =========================================================================
    print(f"\nAnalyzing {len(train_ids)} Visium training samples...")
    
    gene_hvg_votes = defaultdict(int)
    gene_presence_count = defaultdict(int)
    n_train_samples = len(train_ids)
    
    if n_train_samples == 0:
        raise ValueError("No training samples found. Cannot compute gene statistics.")

    for i, sample_id in enumerate(train_ids):
        iterator = iter_hest(data_folder, id_list=[sample_id], load_transcripts=False)
        try:
            st = next(iterator)
            adata = st.adata
            adata = standardize_gene_names(adata)
            sc.pp.filter_genes(adata, min_cells=10)

            for gene in adata.var_names.tolist():
                gene_presence_count[gene] += 1

            sc.pp.highly_variable_genes(
                adata, n_top_genes=3000, flavor='seurat_v3', span=0.3
            )
            for gene in adata.var_names[adata.var['highly_variable']].tolist():
                gene_hvg_votes[gene] += 1

            del st, adata
            gc.collect()
        except Exception as e:
            print(f"  ⚠️ Skipped {sample_id}: {e}")
            continue

    # Build summary dataframe
    presence_series = pd.Series(gene_presence_count)
    vote_series = pd.Series(gene_hvg_votes)

    gene_df = pd.concat([vote_series, presence_series], axis=1).fillna(0).astype(int)
    gene_df.columns = ['hvg_votes', 'presence_count']
    gene_df['visium_fraction'] = gene_df['presence_count'] / n_train_samples
    gene_df['in_xenium'] = gene_df.index.isin(xenium_genes)

    # =========================================================================
    # STEP 3: Select Xenium genes that pass Visium presence filter
    # =========================================================================
    xen_visium_mask = gene_df['in_xenium'] & (gene_df['visium_fraction'] >= vis_presence_thresh)
    selected_xenium_genes = gene_df[xen_visium_mask].index.tolist()

    n_xen_total = len(xenium_genes)
    n_xen_selected = len(selected_xenium_genes)

    print(f"\n--- Xenium → Visium filter ---")
    print(f"  Xenium panel (union): {n_xen_total}")
    print(f"  Present in ≥{vis_presence_thresh*100:.0f}% of Visium training samples: {n_xen_selected}")

    if n_xen_total > n_xen_selected:
        dropped_genes = xenium_genes - set(selected_xenium_genes)
        not_in_visium = [g for g in dropped_genes if g not in gene_df.index]
        below_thresh = [g for g in dropped_genes if g in gene_df.index]
        print(f"  ⚠️ {len(not_in_visium)} Xenium genes not found in any Visium sample")
        print(f"  ⚠️ {len(below_thresh)} Xenium genes below {vis_presence_thresh*100:.0f}% Visium presence")

    # Track which Xenium samples contain each selected gene
    gene_sample_membership = {}
    for gene in selected_xenium_genes:
        samples_with_gene = [sid for sid, genes in per_sample_xenium.items() if gene in genes]
        gene_sample_membership[gene] = samples_with_gene

    # =========================================================================
    # STEP 4: Fill remaining slots with Visium-only HVGs
    # =========================================================================
    remaining_slots = target_count - n_xen_selected
    selected_visium_genes = []

    if remaining_slots > 0:
        filler_mask = (
            ~gene_df.index.isin(selected_xenium_genes)
            & (gene_df['visium_fraction'] >= (1.0 - vis_sparsity_thresh))
        )
        filler_candidates = gene_df[filler_mask].sort_values(
            by=['hvg_votes', 'presence_count'], ascending=[False, False]
        )
        selected_visium_genes = filler_candidates.head(remaining_slots).index.tolist()
        print(f"  Filled with {len(selected_visium_genes)} Visium-only HVGs "
              f"(min presence: {(1.0 - vis_sparsity_thresh)*100:.0f}%)")
    else:
        print(f"  Xenium genes alone meet target ({n_xen_selected} ≥ {target_count}), no fillers needed.")

    # =========================================================================
    # STEP 5: Save outputs
    # =========================================================================
    final_genes_list = selected_xenium_genes + selected_visium_genes
    final_count = len(final_genes_list)

    # --- Main gene list JSON ---
    output_path = Path(data_folder) / f"hybrid_genes_{organ}_{final_count}.json"
    with open(output_path, 'w') as f:
        json.dump(final_genes_list, f, indent=2)

    # --- Detailed stats CSV ---
    final_genes_df = gene_df[gene_df.index.isin(final_genes_list)].copy()
    final_genes_df['source'] = final_genes_df['in_xenium'].map(
        {True: 'xenium', False: 'visium_filler'}
    )
    final_genes_df['xenium_samples'] = final_genes_df.index.map(
        lambda g: ','.join(gene_sample_membership.get(g, []))
    )
    final_genes_df['n_xenium_samples'] = final_genes_df.index.map(
        lambda g: len(gene_sample_membership.get(g, []))
    )

    stats_path = Path(data_folder) / f"hybrid_genes_{organ}_{final_count}_stats.csv"
    final_genes_df[[
        'hvg_votes', 'presence_count', 'visium_fraction',
        'in_xenium', 'source', 'n_xenium_samples', 'xenium_samples'
    ]].to_csv(stats_path)

    # --- Summary ---
    print(f"\nDone. Saved {final_count} genes ({n_xen_selected} Xenium + {len(selected_visium_genes)} Visium filler)")
    print(f"  Gene list: {output_path}")
    print(f"  Stats:     {stats_path}")

    # Per-sample coverage: how many of each sample's panel genes made it in
    if per_sample_xenium:
        print(f"\n--- Per-sample Xenium coverage (of selected Xenium genes) ---")
        for sid in sorted(per_sample_xenium.keys()):
            panel_genes = per_sample_xenium[sid]
            covered = len(set(selected_xenium_genes) & panel_genes)
            print(f"  {sid}: {covered}/{len(panel_genes)} panel genes included for training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, required=True)
    parser.add_argument("--n_genes", type=int, default=5000)
    parser.add_argument("--organ",   type=str, required=True)
    parser.add_argument("--min_xenium_cells_per_gene",   type=int, default=100,
                        help="Min cells expressing a gene for it to be included from Xenium (default: 100)")
    parser.add_argument("--min_xenium_samples_per_gene", type=int, default=1,
                        help="Min number of Xenium samples a gene must pass in (default: 1 = no cross-sample filter)")
    args = parser.parse_args()

    generate_hybrid_gene_list(
        args.config, args.n_genes, args.organ,
        min_xenium_cells_per_gene=args.min_xenium_cells_per_gene,
        min_xenium_samples_per_gene=args.min_xenium_samples_per_gene,
    )