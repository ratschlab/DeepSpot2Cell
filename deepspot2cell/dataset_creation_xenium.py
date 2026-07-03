import os
import gc
import json
import h5py
import pyvips
import pandas as pd
from shapely.geometry import box, Polygon, shape, Point
import argparse
from shapely.ops import unary_union
import numpy as np
from hest import iter_hest
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import time
import psutil
from contextlib import nullcontext
import scanpy as sc
from pathlib import Path
from scipy.spatial import cKDTree

from deepspot2cell.utils.utils import load_config, order_genes, standardize_gene_names, clean_gene_string, is_artifact
from deepspot2cell.utils.utils_image import format_to_dtype, get_morphology_model_and_preprocess, check_patch_quality

# --- DEBUG UTILITIES ---
DEBUG_MODE = False

def debug_print(*args, **kwargs):
    """Helper to print only if DEBUG_MODE is True."""
    if DEBUG_MODE:
        print(*args, **kwargs)

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
# -----------------------

def process_batch(batch_patches, model, preprocess, device):
    batch_processed = []
    for patch in batch_patches:
        patch_img = Image.fromarray(patch)
        if patch_img.width != patch_img.height:
            size = max(patch_img.width, patch_img.height)
            patch_img = patch_img.resize((size, size))
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        batch_processed.append(preprocess(patch_img).to(device).to(dtype))

    batch_tensor = torch.stack(batch_processed)

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    with autocast_ctx:
        with torch.inference_mode():
            batch_embeddings = model(batch_tensor)
            batch_embeddings = batch_embeddings.detach().cpu()

    return batch_embeddings


def get_spot_embs(X_spot, morphology_model, preprocess, device, dtype_to_use=torch.float16):
    dtype_to_use = torch.float16 if device.type == "cuda" else torch.float32
    autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype_to_use) if device.type == "cuda" else nullcontext()
    with autocast_ctx:
        with torch.inference_mode():
            X_spot_img = Image.fromarray(X_spot)
            # if spot is not square, make it square
            if X_spot_img.width != X_spot_img.height:
                size = max(X_spot_img.width, X_spot_img.height)
                X_spot_img = X_spot_img.resize((size, size))

            X_spot_tensor = preprocess(X_spot_img).to(device).to(dtype_to_use)
            token_map = morphology_model(X_spot_tensor[None, ])
            cls_tok = token_map.squeeze(0)

        return cls_tok.cpu().numpy()


def load_master_gene_list(adata, config):
    ordered_genes_path = f"{config['data']['data_folder']}/{config['data']['ordered_genes_file']}"
    print(f"  Using gene order from {ordered_genes_path}")

    if os.path.exists(ordered_genes_path):
        print(f"  Loading ordered genes from {ordered_genes_path}")
    else:
        print(f"  Ordered genes file not found. Generating new order.")
        order_genes(config)

    with open(ordered_genes_path, 'r') as f:
        ordered_gene_names = json.load(f)


    adata_genes_set = set(adata.var.index.values)
    ordered_genes_set = set(ordered_gene_names)

    intersection = adata_genes_set.intersection(ordered_genes_set)
    print(f"DEBUG: Xenium Sample has {len(adata_genes_set)} genes.")
    print(f"DEBUG: Master list has {len(ordered_genes_set)} genes.")
    print(f"DEBUG: Overlap count: {len(intersection)}")
    print(f"DEBUG: First 30 Xenium genes: {list(adata_genes_set)[:30]}")
    print(f"DEBUG: First 30 Master genes: {list(ordered_genes_set)[:30]}")

    if adata_genes_set != ordered_genes_set:
        print("  Warning: Gene names in adata and ordered genes file do not match.")
        diff = adata_genes_set - ordered_genes_set
        if len(diff) > 0:
            print(f"  Different genes in adata: {diff}")

    return ordered_gene_names


def crop_tile(image, x_pixel, y_pixel, spot_diameter):
    spot = image.crop(x_pixel, y_pixel, spot_diameter, spot_diameter)
    main_tile = np.ndarray(buffer=spot.write_to_memory(),
                           dtype=format_to_dtype[spot.format],
                           shape=[spot.height, spot.width, spot.bands])
    main_tile = main_tile[:, :, :3]
    return main_tile


def fix_invalid_geometry(geom):
    """Attempt to fix invalid geometry"""
    if not geom.is_valid:
        try:
            fixed = geom.buffer(0)
            if fixed.is_valid:
                return fixed

            if hasattr(geom, 'make_valid'):
                fixed = geom.make_valid()
                if fixed.is_valid:
                    return fixed
        except Exception:
            pass
    return geom


def create_tissue_mask(tissue_mask_json, sample_id):
    """Create a valid tissue mask from GeoJSON, handling geometry errors robustly"""
    tissue_polygons = []
    total_polygons = 0
    invalid_polygons = 0

    for feature in tissue_mask_json['features']:
        if feature['geometry']['type'] != 'Polygon':
            continue
        total_polygons += 1

        try:
            poly = shape(feature['geometry'])
            if not poly.is_valid:
                poly = poly.buffer(0)
                if not poly.is_valid:
                    invalid_polygons += 1
                    continue
            tissue_polygons.append(poly)
        except Exception:
            invalid_polygons += 1

    if not tissue_polygons:
        print(f"  Warning: No valid polygons in {sample_id} (total: {total_polygons}, invalid: {invalid_polygons})")
        return None
    print(f"  Extracted {len(tissue_polygons)} valid polygons from {total_polygons} total")
    if len(tissue_polygons) == 1:
        return tissue_polygons[0]

    try:
        return unary_union(tissue_polygons)
    except Exception:
        pass

    mask = tissue_polygons[0]
    successful_unions = 1

    for i, poly in enumerate(tissue_polygons[1:], 1):
        try:
            clean_mask = mask.buffer(0)
            clean_poly = poly.buffer(0)
            if clean_mask.is_valid and clean_poly.is_valid:
                mask = clean_mask.union(clean_poly)
                successful_unions += 1
            else:
                print(f"  Warning: Skipping polygon {i} in union due to validity issues")
        except Exception:
            print(f"  Warning: Failed to union polygon {i}, continuing with current mask")
    print(f"  Created tissue mask with {successful_unions}/{len(tissue_polygons)} polygons")
    return mask


def process_sample(sample_id, st, model, preprocess, device, batch_size, config):
    data_folder = config['data']['data_folder']
    dataset_variant = config['data']['dataset_variant']
    print(f"Processing {sample_id}")
    print_system_status(f"Starting Sample {sample_id}")
    
    if os.path.exists(f'{data_folder}/expressions{dataset_variant}/{sample_id}_expressions.h5'):
        print(f"  Already processed {sample_id}, skipping")
        return

    exp_name = config['experiment']['name']
    base_output_dir = config['experiment']['output_dir']
    stats_dir = Path(base_output_dir) / exp_name / "preprocessing_stats"
    os.makedirs(stats_dir, exist_ok=True)
    stats_file_path = stats_dir / f"{sample_id}_stats.json"

    with open(f"{data_folder}/metadata/{sample_id}.json") as f:
        metadata = json.load(f)
    
    # --- DYNAMIC RESOLUTION SETUP ---
    um_per_px = metadata.get('pixel_size_um_estimated', 0.2125)
    
    target_input_px = config['data'].get('target_input_size_px', 224)
    target_context_um = config['data'].get('target_context_um', 112.0)
    target_min_cell_um = config['data'].get('min_cell_context_um', 36.0)
    spot_diameter_um = config['data'].get('spot_diameter_um', 55.0)

    patch_size = int(target_context_um / um_per_px)
    spot_diameter_px = int(spot_diameter_um / um_per_px)
    min_cell_crop_px = int(target_min_cell_um / um_per_px)

    print(f"  Resolution Normalization:")
    print(f"    - Resolution: {um_per_px:.4f} um/px")
    print(f"    - Patch Size: {patch_size} px ({target_context_um} um)")
    print(f"    - Spot Dia:   {spot_diameter_px} px ({spot_diameter_um} um)")
    print(f"    - Min Cell:   {min_cell_crop_px} px ({target_min_cell_um} um)")

    print("  Standardizing gene names in AnnData...")
    st.adata = standardize_gene_names(st.adata)
    
    valid_genes_mask = [not is_artifact(g) for g in st.adata.var_names]
    st.adata = st.adata[:, valid_genes_mask].copy()

    ordered_gene_names = load_master_gene_list(st.adata, config)
    
    master_set = set(ordered_gene_names)
    sample_set = set(st.adata.var.index)
    common = master_set.intersection(sample_set)
    overlap_pct = (len(common) / len(master_set)) * 100 if len(master_set) > 0 else 0.0

    gene_to_idx = {gene: idx for idx, gene in enumerate(ordered_gene_names)}

    with open(f"{data_folder}/tissue_seg/{sample_id}_contours.geojson", 'r') as f:
        tissue_mask_json = json.load(f)
    tissue_mask = create_tissue_mask(tissue_mask_json, sample_id)

    cell_shapes = st.get_shapes('xenium_nucleus', 'he').shapes
    if cell_shapes.index.dtype == 'object':
        decode = lambda x: x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x)
        cell_shapes.index = cell_shapes.index.map(decode).astype("string")

    print_system_status("Loading WSI to RAM")
    wsi = pyvips.Image.new_from_file(f"{data_folder}/wsis/{sample_id}.tif")
    wsi_width = wsi.width
    wsi_height = wsi.height

    # --- TRANSCRIPT FILTERING & SPATIAL TREE SETUP ---
    # 1. Start with all transcripts to capture extracellular/ambient RNA
    all_tx_df = st.transcript_df.copy()

    qv_thr = config['data']['qv_thr']
    initial_transcript_count = len(all_tx_df)
    all_tx_df['qv'] = pd.to_numeric(all_tx_df['qv'], errors='coerce')
    all_tx_df = all_tx_df[all_tx_df['qv'] > qv_thr]
    print(f"  Filtered by 'qv' > {qv_thr}: {len(all_tx_df)} transcripts retained from {initial_transcript_count}.")

    for col in ('feature_name', 'cell_id'):
        if col in all_tx_df.columns:
            s = all_tx_df[col]
            if s.dtype == 'object':
                all_tx_df[col] = s.str.decode('utf-8', errors='ignore').fillna(s).astype('string')

    all_tx_df['feature_name'] = all_tx_df['feature_name'].apply(clean_gene_string)

    # --- Mark noise vs biological transcripts (keep both so we can compute noise fraction per cell) ---
    all_tx_df['is_noise'] = all_tx_df['feature_name'].apply(is_artifact)
    noise_tx_count = int(all_tx_df['is_noise'].sum())
    print(f"  Transcripts: {len(all_tx_df) - noise_tx_count} biological, "
          f"{noise_tx_count} noise/control probes retained for noise-fraction QC.")

    # Work only with biological transcripts for gene indexing and KDTree
    bio_tx_df = all_tx_df[~all_tx_df['is_noise']].copy()

    print_system_status("Building Spatial Transcript Tree")

    # 2. Map genes to their exact index in our master gene list
    bio_tx_df['gene_idx'] = bio_tx_df['feature_name'].map(gene_to_idx)

    # Drop transcripts belonging to genes outside our master list
    bio_tx_df = bio_tx_df.dropna(subset=['gene_idx'])
    bio_tx_df['gene_idx'] = bio_tx_df['gene_idx'].astype(np.int32)

    # 3. Extract coordinates and gene indices as raw numpy arrays (Highly memory efficient)
    tx_coords = bio_tx_df[['he_x', 'he_y']].values.astype(np.float32)
    tx_gene_indices = bio_tx_df['gene_idx'].values

    # 4. Build the KDTree for instant geometric radius queries later
    print("  Building cKDTree for spot expression queries...")
    tx_kdtree = cKDTree(tx_coords)

    # 5. Cell-assigned biological transcripts for expression aggregation
    filtered_df = bio_tx_df[~bio_tx_df.cell_id.isin(['UNASSIGNED', '-1', -1])].copy()

    # Cell-assigned noise transcripts — used only to compute per-cell noise fraction
    noise_cell_df = all_tx_df[
        all_tx_df['is_noise'] & ~all_tx_df.cell_id.isin(['UNASSIGNED', '-1', -1])
    ].copy()

    # Free the massive initial DataFrames
    del all_tx_df, bio_tx_df
    st.transcript_df = None
    gc.collect()

    print_system_status("Pre-Aggregation")
    # --- MEMORY AND SPEED OPTIMIZATION START ---
    # 1. Fast global aggregation (avoids looping over a 104M row groupby object)
    print("  Aggregating biological transcript counts globally...")
    cell_gene_counts = filtered_df.groupby(['cell_id', 'feature_name']).size().reset_index(name='count')

    # Aggregate noise transcript counts per cell (total, for noise-fraction QC)
    print("  Aggregating noise transcript counts per cell...")
    if len(noise_cell_df) > 0:
        cell_noise_dict = noise_cell_df.groupby('cell_id').size().to_dict()
    else:
        cell_noise_dict = {}

    # 2. Delete massive DataFrames to free ~15-20GB RAM immediately
    del filtered_df, noise_cell_df
    st.transcript_df = None
    gc.collect()
    print_system_status("Post-Aggregation & GC")

    print("  Building cell expression vectors...")
    cell_expr_dict = {}
    for row in cell_gene_counts.itertuples(index=False):
        c_id = row.cell_id
        g_name = row.feature_name

        if g_name in gene_to_idx:
            if c_id not in cell_expr_dict:
                # Use float32 to halve array RAM consumption (saves ~25GB)
                cell_expr_dict[c_id] = np.zeros(len(ordered_gene_names), dtype=np.float32)
            cell_expr_dict[c_id][gene_to_idx[g_name]] = row.count

    del cell_gene_counts
    gc.collect()

    # --- Cell Quality Filters ---
    min_cell_counts    = config['data'].get('min_cell_counts',    15)
    min_genes_per_cell = config['data'].get('min_genes_per_cell', 10)
    max_cell_counts    = config['data'].get('max_cell_counts',    None)  # None = auto 99.5th pct
    max_noise_fraction = config['data'].get('max_noise_fraction', 0.05)

    # Auto-compute max_counts threshold (99.5th percentile of this sample) if not set
    if max_cell_counts is None:
        _all_counts = np.array([np.sum(e) for e in cell_expr_dict.values()], dtype=np.float32)
        max_counts_threshold = float(np.percentile(_all_counts, 99.5))
        del _all_counts
    else:
        max_counts_threshold = float(max_cell_counts)

    pre_filter_count      = len(cell_expr_dict)
    n_removed_min_counts  = 0
    n_removed_min_genes   = 0
    n_removed_max_counts  = 0
    n_removed_noise_frac  = 0

    filtered_cell_expr = {}
    for cid, expr in cell_expr_dict.items():
        total_counts = float(np.sum(expr))
        n_genes      = int(np.count_nonzero(expr))
        noise_counts = float(cell_noise_dict.get(cid, 0))
        noise_frac   = noise_counts / (total_counts + noise_counts) if (total_counts + noise_counts) > 0 else 0.0

        if total_counts < min_cell_counts:
            n_removed_min_counts += 1; continue
        if n_genes < min_genes_per_cell:
            n_removed_min_genes += 1; continue
        if total_counts > max_counts_threshold:
            n_removed_max_counts += 1; continue
        if noise_frac > max_noise_fraction:
            n_removed_noise_frac += 1; continue
        filtered_cell_expr[cid] = expr

    cell_expr_dict = filtered_cell_expr
    del filtered_cell_expr, cell_noise_dict
    gc.collect()

    post_filter_count = len(cell_expr_dict)
    print(f"  Cell QC filter summary (input: {pre_filter_count} cells):")
    print(f"    Removed min_counts   < {min_cell_counts}:      {n_removed_min_counts}")
    print(f"    Removed min_genes    < {min_genes_per_cell}:      {n_removed_min_genes}")
    print(f"    Removed max_counts   > {max_counts_threshold:.0f}:   {n_removed_max_counts}")
    print(f"    Removed noise_frac   > {max_noise_fraction*100:.0f}%:    {n_removed_noise_frac}")
    print(f"    Remaining cells:         {post_filter_count}")

    # 4. Create O(1) dictionary lookup for geometry (fixes the 4-hour ETA bottleneck)
    print("  Building fast geometry lookup...")
    cell_geom_dict = dict(zip(cell_shapes.index, cell_shapes.geometry))

    # --- Area-based cell QC parameters (applied inside the cell loop) ---
    min_cell_area_um2 = config['data'].get('min_cell_area_um2', 6.0)
    max_cell_area_um2 = config['data'].get('max_cell_area_um2', 400.0)  # aligned with Visium default

    cell_dict = {}
    cell_batch = []
    cell_batch_ids = []

    # We now get a 100% accurate total count, making ETAs perfect
    valid_cell_ids = list(cell_expr_dict.keys())
    total_cells_to_process = len(valid_cell_ids)
    total_cells = 0
    cells_inside_tissue   = 0
    cells_filtered_area_small = 0
    cells_filtered_area_large = 0

    cell_start_time = time.time()
    print(f"  Starting processing of {total_cells_to_process} cells for {sample_id}")

    for cell_id in valid_cell_ids:
        total_cells += 1
        if total_cells % 10000 == 0:
            elapsed = time.time() - cell_start_time
            cells_per_sec = total_cells / elapsed
            remaining = (total_cells_to_process - total_cells) / cells_per_sec if cells_per_sec > 0 else 0
            print(f"  Processed {total_cells}/{total_cells_to_process} cells ({total_cells/total_cells_to_process*100:.1f}%) - "
                  f"ETA: {int(elapsed/3600)}:{int((elapsed%3600)/60):02d}:{int(elapsed%60):02d} / "
                  f"{int(remaining/3600)}:{int((remaining%3600)/60):02d}:{int(remaining%60):02d}")
            print_system_status(f"Cell Processing Loop {total_cells}")

        gene_expression = cell_expr_dict[cell_id]

        # Instant O(1) lookup instead of iterating through 1.3M rows
        cell_shape = cell_geom_dict.get(cell_id)
        if cell_shape is None:
            continue

        if not cell_shape.is_valid:
            cell_shape = fix_invalid_geometry(cell_shape)
            if not cell_shape.is_valid:
                print(f"  Cell {cell_id} has invalid geometry after fixing")
                continue
        
        if cell_shape.is_empty:
            # print(f"  Cell {cell_id} is an empty geometry, skipping.")
            continue
                
        centroid = cell_shape.centroid

        if centroid.is_empty:
            continue

        cell_centroid_x = centroid.x
        cell_centroid_y = centroid.y

        # Check if cell is inside tissue mask
        if not tissue_mask.contains(Point(cell_centroid_x, cell_centroid_y)):
            continue

        cells_inside_tissue += 1
        cell_area = cell_shape.area

        # Area bounds filter: removes debris fragments (too small) and merged-cell artifacts (too large)
        cell_area_um2 = cell_area * (um_per_px ** 2)
        if cell_area_um2 < min_cell_area_um2:
            cells_filtered_area_small += 1
            continue
        if max_cell_area_um2 is not None and cell_area_um2 > max_cell_area_um2:
            cells_filtered_area_large += 1
            continue

        cell_bounds = cell_shape.bounds

        rect_width = cell_bounds[2] - cell_bounds[0]
        rect_height = cell_bounds[3] - cell_bounds[1]
        
        crop_size = max(rect_width, rect_height, min_cell_crop_px)
        max_dimension = int(np.ceil(crop_size))

        # Safely convert to integer pixel coordinates
        square_minx = int(round(cell_centroid_x - max_dimension / 2.0))
        square_miny = int(round(cell_centroid_y - max_dimension / 2.0))
        square_maxx = square_minx + max_dimension
        square_maxy = square_miny + max_dimension

        try:
            cell_img = crop_tile(wsi, square_minx, square_miny, max_dimension)
        except Exception as e:
            print(f"  Error cropping cell {cell_id} image: {e}")
            continue

        # EXPLICIT RESIZE (To 224x224)
        cell_img_pil = Image.fromarray(cell_img)
        if cell_img_pil.size != (target_input_px, target_input_px):
            cell_img_pil = cell_img_pil.resize((target_input_px, target_input_px), resample=Image.LANCZOS)
        cell_img = np.array(cell_img_pil)

        cell_dict[f"{sample_id}_{cell_id}"] = {
            'cell_expression': gene_expression,
            'cell_shape': cell_shape,
            'cell_centroid_x': cell_centroid_x,
            'cell_centroid_y': cell_centroid_y,
            'area': cell_area,
            'cell_bounds': cell_bounds,
            'square_bbox_coords': (square_minx, square_miny, square_maxx, square_maxy),
        }

        cell_batch.append(cell_img)
        cell_batch_ids.append(f"{sample_id}_{cell_id}")
        if len(cell_batch) == batch_size:
            cell_embeddings = process_batch(cell_batch, model, preprocess, device)
            for j, c_id in enumerate(cell_batch_ids):
                cell_dict[c_id]['cell_embedding'] = cell_embeddings[j]
            cell_batch = []
            cell_batch_ids = []

    if len(cell_batch) > 0:
        cell_embeddings = process_batch(cell_batch, model, preprocess, device)
        for j, c_id in enumerate(cell_batch_ids):
            cell_dict[c_id]['cell_embedding'] = cell_embeddings[j]

    # Free the expression & geometry dicts as their data is now stored safely inside cell_dict
    del cell_expr_dict
    del cell_geom_dict
    gc.collect()

    elapsed = time.time() - cell_start_time
    print(f"  Completed cell processing: {cells_inside_tissue}/{total_cells} cells inside tissue in {elapsed:.1f} sec")
    print(f"  Area QC: removed {cells_filtered_area_small} too-small (<{min_cell_area_um2} μm²), "
          f"{cells_filtered_area_large} too-large (>{max_cell_area_um2} μm²) cells")

    patches_outside_tissue = 0
    patches_visual_filtered = 0
    patches_no_cells = 0
    patches_low_count = 0
    patches_inside_tissue = 0
    spot_dict = {}

    # --- VISIUM-MIMIC HEXAGONAL GRID SETUP ---
    # Visium center-to-center distance is exactly 100 um
    c2c_um = 100.0
    c2c_px = c2c_um / um_per_px
    
    # In a hex grid, row height is c2c * sqrt(3)/2
    row_spacing_px = c2c_px * (np.sqrt(3) / 2.0)
    col_spacing_px = c2c_px

    # Calculate number of rows and columns that safely fit in the WSI
    # We subtract patch_size to account for the margin padding we add to centers
    num_rows = int(max(0, wsi_height - patch_size) / row_spacing_px)
    num_cols = int(max(0, wsi_width - patch_size) / col_spacing_px)

    # --- BUILD CELL KDTREE FOR FAST SEARCH ---
    print("  Building cKDTree for cell spatial queries...")
    valid_cell_ids_list = list(cell_dict.keys())
    # Extract centroids to a fast numpy array
    cell_centroids = np.array([
        [cell_dict[cid]['cell_centroid_x'], cell_dict[cid]['cell_centroid_y']] 
        for cid in valid_cell_ids_list
    ])
    cell_kdtree = cKDTree(cell_centroids)
    # -----------------------------------------

    total_patch_count = num_rows * num_cols
    patch_count = 0
    patches_saved = 0
    unique_saved_cells = set()
    total_cell_patch_assignments = 0
    print(f"  Starting processing of {total_patch_count} hex-grid patches...")
    patch_start_time = time.time()

    h5_output_path = f'{data_folder}/expressions{dataset_variant}/{sample_id}_expressions.h5'
    
    # Open the H5 file BEFORE the loop to write incrementally
    with h5py.File(h5_output_path, 'w') as h5_file:
        for row in range(num_rows):
            for col_idx in range(num_cols):
                patch_count += 1
                if patch_count % 1000 == 0:
                    elapsed = time.time() - patch_start_time
                    patches_per_sec = patch_count / elapsed
                    remaining = (total_patch_count - patch_count) / patches_per_sec if patches_per_sec > 0 else 0
                    print(f"  Processed {patch_count}/{total_patch_count} patches ({patch_count/total_patch_count*100:.1f}%) - "
                          f"ETA: {int(remaining/60)} min {int(remaining)%60} sec")
                    print_system_status(f"Patch Processing Loop {patch_count}")

                col = col_idx * 2 + (row % 2)
                patch_barcode = f"patch_{row}_{col}_{sample_id}"

                patch_center_x = int(round(col * (c2c_px / 2.0) + (patch_size / 2.0)))
                patch_center_y = int(round(row * row_spacing_px + (patch_size / 2.0)))
                patch_x = int(round(patch_center_x - patch_size / 2.0))
                patch_y = int(round(patch_center_y - patch_size / 2.0))

                if patch_x < 0 or patch_y < 0 or patch_x + patch_size > wsi_width or patch_y + patch_size > wsi_height:
                    continue

                if not tissue_mask.contains(Point(patch_center_x, patch_center_y)):
                    patches_outside_tissue += 1
                    continue

                patches_inside_tissue += 1
                
                # --- FAST KD-TREE QUERY ---
                # Find cell indices within the search radius instantly
                nearby_idx = cell_kdtree.query_ball_point([patch_center_x, patch_center_y], r=patch_size, p=np.inf)
                nearby_cells = {valid_cell_ids_list[i]: cell_dict[valid_cell_ids_list[i]] for i in nearby_idx}
                
                if len(nearby_cells) == 0:
                    patches_no_cells += 1
                    continue

                try:
                    patch_img = crop_tile(wsi, patch_x, patch_y, patch_size)
                except Exception as e:
                    continue

                patch_img_pil = Image.fromarray(patch_img)
                if patch_img_pil.size != (target_input_px, target_input_px):
                    patch_img_pil = patch_img_pil.resize((target_input_px, target_input_px), resample=Image.LANCZOS)
                patch_img = np.array(patch_img_pil)

                if not check_patch_quality(patch_img):
                    patches_visual_filtered += 1
                    continue

                patch_box = box(patch_x, patch_y, patch_x + patch_size, patch_y + patch_size)
                cell_expressions = []
                cell_bboxes_adjusted = []
                cell_centroids_adjusted = []
                cell_ids = []
                cell_ids_tagged = []

                if not patch_box.is_valid:
                    patch_box = fix_invalid_geometry(patch_box)
                    if not patch_box.is_valid:
                        continue

                spot_radius = spot_diameter_px // 2
                spot = Point(patch_center_x, patch_center_y).buffer(spot_radius, cap_style=1)

                for cell_id, cell_data in nearby_cells.items():
                    cell_shape = cell_data['cell_shape']
                    cell_bounds = cell_data['cell_bounds']
                    patch_bounds = patch_box.bounds 

                    if (cell_bounds[2] < patch_bounds[0] or cell_bounds[0] > patch_bounds[2] or
                        cell_bounds[3] < patch_bounds[1] or cell_bounds[1] > patch_bounds[3]):
                        continue

                    if not (cell_bounds[0] >= patch_bounds[0] and cell_bounds[2] <= patch_bounds[2] and
                        cell_bounds[1] >= patch_bounds[1] and cell_bounds[3] <= patch_bounds[3]):
                        if cell_shape.intersects(patch_box):
                            intersection = cell_shape.intersection(patch_box)
                            if intersection.area / cell_data['area'] < 0.6:
                                continue
                        else:
                            continue

                    if spot.contains(Point(cell_data['cell_centroid_x'], cell_data['cell_centroid_y'])):
                        cell_id_tagged = f"{cell_id}_in"
                    else:
                        cell_id_tagged = f"{cell_id}_out"

                    cell_expressions.append(cell_data['cell_expression'])
                    cell_ids.append(cell_id)
                    cell_ids_tagged.append(cell_id_tagged)
                    
                    cell_bbox_coords = cell_data['square_bbox_coords']
                    cell_bboxes_adjusted.append((
                        (cell_bbox_coords[0] - patch_x) / patch_size,
                        (cell_bbox_coords[1] - patch_y) / patch_size,
                        (cell_bbox_coords[2] - patch_x) / patch_size,
                        (cell_bbox_coords[3] - patch_y) / patch_size
                    ))
                    
                    cell_centroids_adjusted.append((
                        (cell_data['cell_centroid_x'] - patch_x) / patch_size,
                        (cell_data['cell_centroid_y'] - patch_y) / patch_size
                    ))

                if len(cell_ids) == 0:
                    patches_no_cells += 1
                    continue

                cell_expressions = np.array(cell_expressions)
                
                spot_radius_px = spot_diameter_px / 2.0
                tx_indices_in_spot = tx_kdtree.query_ball_point([patch_center_x, patch_center_y], r=spot_radius_px)
                
                if len(tx_indices_in_spot) > 0:
                    genes_in_spot = tx_gene_indices[tx_indices_in_spot]
                    gene_counts = np.bincount(genes_in_spot, minlength=len(ordered_gene_names))
                    spot_expression = gene_counts.astype(np.float32)
                else:
                    spot_expression = np.zeros(len(ordered_gene_names), dtype=np.float32)

                if np.sum(spot_expression) < 10:
                    patches_low_count += 1
                    continue

                unique_saved_cells.update(cell_ids)
                total_cell_patch_assignments += len(cell_ids)

                # --- INCREMENTAL HDF5 WRITE ---
                spot_group = h5_file.create_group(patch_barcode)
                spot_group.attrs['spot_diameter'] = spot_diameter_px
                spot_group.attrs['patch_size'] = patch_size
                spot_group.attrs['um_per_px'] = um_per_px
                
                spot_group.create_dataset('spot_expression', data=spot_expression, compression='gzip', compression_opts=4)
                spot_group.create_dataset('cell_expressions', data=cell_expressions, compression='gzip', compression_opts=4)
                
                cell_ids_array = np.array(cell_ids_tagged, dtype=h5py.string_dtype())
                spot_group.create_dataset('cell_ids', data=cell_ids_array)
                spot_group.create_dataset('cell_centroids', data=np.array(cell_centroids_adjusted, dtype=np.float32), compression='gzip', compression_opts=4)
                spot_group.create_dataset('cell_bbxs', data=np.array(cell_bboxes_adjusted, dtype=np.float32), compression='gzip', compression_opts=4)
                
                # Flush every 500 patches to guarantee memory is freed
                patches_saved += 1
                if patches_saved % 500 == 0:
                    h5_file.flush()
                # ------------------------------

                spot_embeddings = get_spot_embs(patch_img, model, preprocess, device)
                patch_embeddings = [spot_embeddings.astype(np.float16)]
                for cell_id in cell_ids:
                    emb = cell_dict[cell_id]['cell_embedding']
                    if torch.is_tensor(emb):
                        emb = emb.numpy()
                    patch_embeddings.append(emb.astype(np.float16))
                    
                patch_embeddings = np.vstack(patch_embeddings)
                np.save(f'{data_folder}/embeddings{dataset_variant}/{model_name}/{patch_barcode}.npy', patch_embeddings)

    elapsed_total = time.time() - patch_start_time
        
    print(f"\n--- Processing Summary for Xenium {sample_id} ---")
    print(f"  Time Elapsed:          {elapsed_total:.1f} sec")
    print(f"  Total Grid Patches:    {total_patch_count}")
    print(f"  Spots outside Tissue:  {patches_outside_tissue}")
    print(f"  Visual QC Filtered:    {patches_visual_filtered}")
    print(f"  No Cells nearby/valid: {patches_no_cells}")
    print(f"  Low Expression (<10):  {patches_low_count}")
    print(f"  TOTAL SAVED PATCHES:   {patches_saved}")
    print(f"  TOTAL UNIQUE CELLS:    {len(unique_saved_cells)}")
    print(f"----------------------------------------------\n")

    print(f"  Saved spot data incrementally to {data_folder}/expressions{dataset_variant}/{sample_id}_expressions.h5")

    stats_data = {
        "sample_id": sample_id,
        "resolution_um_px": float(um_per_px),
        "gene_overlap_pct": round(overlap_pct, 2),
        "initial_spots": int(total_patch_count),
        "low_count_filtered": int(patches_low_count),
        "spots_outside_tissue": int(patches_outside_tissue),
        "visual_qc_filtered": int(patches_visual_filtered),
        "no_cells_nearby": int(patches_no_cells),
        "final_saved_patches": patches_saved,

        "total_cells_segmented": int(len(cell_shapes)),
        "cells_with_transcripts": int(pre_filter_count),
        "cells_inside_tissue": int(cells_inside_tissue),
        "valid_cells_embedded": int(len(cell_dict)),
        "unique_cells_in_saved_patches": int(len(unique_saved_cells)),
        "total_cell_patch_assignments": int(total_cell_patch_assignments),

        # --- Cell QC filter statistics ---
        "qc_thresholds": {
            "min_cell_counts":    min_cell_counts,
            "min_genes_per_cell": min_genes_per_cell,
            "max_cell_counts":    round(max_counts_threshold, 1),
            "max_noise_fraction": max_noise_fraction,
            "min_cell_area_um2":  min_cell_area_um2,
            "max_cell_area_um2":  max_cell_area_um2,
        },
        "qc_cells_pre_filter":         int(pre_filter_count),
        "qc_cells_post_filter":        int(post_filter_count),
        "qc_removed_min_counts":       int(n_removed_min_counts),
        "qc_removed_min_genes":        int(n_removed_min_genes),
        "qc_removed_max_counts":       int(n_removed_max_counts),
        "qc_removed_noise_frac":       int(n_removed_noise_frac),
        "qc_removed_area_small":       int(cells_filtered_area_small),
        "qc_removed_area_large":       int(cells_filtered_area_large),

        "is_xenium": 1.
    }
    try:
        with open(stats_file_path, 'w') as f:
            json.dump(stats_data, f, indent=4)
        print(f"  Saved statistics to {stats_file_path}")
    except Exception as e:
        print(f"  WARNING: Failed to save statistics json: {e}")

    del cell_dict
    gc.collect()
    print_system_status("Post-Sample Cleanup")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Xenium slides')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config file')
    parser.add_argument('--split', type=str, required=True, help='Split name to process (e.g., test)')
    parser.add_argument('--indices', type=int, nargs='+', help='List of specific indices to process')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug logging and memory profiling')
    args = parser.parse_args()

    DEBUG_MODE = args.debug

    print_system_status("Startup")

    config = load_config(args.config)

    data_cfg = config['data']
    data_folder = data_cfg['data_folder']
    model_name = data_cfg['model_name']
    model_path = data_cfg['model_path']
    dataset_variant = data_cfg['dataset_variant']

    if args.split not in data_cfg['split']:
        raise ValueError(f"Split '{args.split}' not found in config")

    sample_ids = data_cfg['split'][args.split]['ids']

    if args.indices:
        sample_ids = [sample_ids[i] for i in args.indices]

    ids_to_process = []
    print(f"Checking status of {len(sample_ids)} samples...")
    for s_id in sample_ids:
        expected_file = f'{data_folder}/expressions{dataset_variant}/{s_id}_expressions.h5'
        if os.path.exists(expected_file):
            print(f"  [Skipping] {s_id} - Already processed.")
        else:
            ids_to_process.append(s_id)

    if not ids_to_process:
        print("All requested samples are already processed. Exiting.")
        exit(0)

    print(f"Processing remaining {len(ids_to_process)} samples: {ids_to_process}")

    processing_cfg = config.get('processing', {})
    if torch.cuda.is_available():
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024**3)
        auto_batch = int(vram_gb * 32 / 14) if vram_gb > 13 else 16
        batch_size = processing_cfg.get('gpu_batch_size', auto_batch)
        device = torch.device('cuda')
        print(f"Using CUDA with VRAM {vram_gb:.1f} GB; batch size set to {batch_size}")
    else:
        batch_size = processing_cfg.get('cpu_batch_size', 8)
        device = torch.device('cpu')
        print(f"CUDA not available; falling back to CPU with batch size {batch_size}")

    print_system_status("Pre-Model Load")
    model, preprocess, feature_dim = get_morphology_model_and_preprocess(model_name, device.type, model_path=model_path)
    model.to(device)
    print_system_status("Model Loaded")

    os.makedirs(f'{data_folder}/embeddings{dataset_variant}/', exist_ok=True)
    os.makedirs(f'{data_folder}/embeddings{dataset_variant}/{model_name}', exist_ok=True)
    os.makedirs(f'{data_folder}/expressions{dataset_variant}', exist_ok=True)

    for i, st in enumerate(iter_hest(data_folder, id_list=ids_to_process, load_transcripts=True)):
        sample_id = ids_to_process[i]
        process_sample(sample_id, st, model, preprocess, device, batch_size, config)
