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
from contextlib import nullcontext, contextmanager
from collections import defaultdict
import scanpy as sc
from pathlib import Path
import geopandas as gpd
import psutil
from scipy.spatial import cKDTree
import pyogrio
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from deepspot2cell.utils.utils import load_config, order_genes, standardize_gene_names
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


def load_master_gene_list(adata_var_index, config):
    """
    Loads the pre-computed master gene list (JSON) and creates an index mapping.
    Verifies how well the current sample matches the master list.
    """
    filename = config['data'].get('ordered_genes_file')
    if not filename:
        raise ValueError("Config missing 'data.ordered_genes_file' key!")
        
    ordered_genes_path = os.path.join(config['data']['data_folder'], filename)
    print(f"  Loading master gene order from: {filename}")

    if not os.path.exists(ordered_genes_path):
        raise FileNotFoundError(f"Master gene list not found at: {ordered_genes_path}")

    with open(ordered_genes_path, 'r') as f:
        ordered_gene_names = json.load(f)

    gene_to_ordered_idx = {gene: idx for idx, gene in enumerate(ordered_gene_names)}

    adata_genes_set = set(adata_var_index)
    master_genes_set = set(ordered_gene_names)
    
    common_genes = master_genes_set.intersection(adata_genes_set)
    overlap_pct = (len(common_genes) / len(master_genes_set)) * 100

    print(f"    Sample/Master Overlap: {len(common_genes)}/{len(master_genes_set)} genes ({overlap_pct:.1f}%)")
    
    if overlap_pct < 50.0:
        print(f"    WARNING: Low gene overlap detected!")

    return ordered_gene_names, gene_to_ordered_idx


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

def save_spot_visualization(patch_img, patch_x, patch_y, patch_size, target_input_px,
                             spot_radius, patch_center_x, patch_center_y,
                             cell_ids, cell_ids_tagged, cell_dict,
                             cells_in_spot, sample_id, patch_barcode, debug_dir):
    """Save a debug visualization of one spot: H&E patch + Visium circle + CellViT segmentations."""
    scale = target_input_px / patch_size

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(patch_img)

    # Draw Visium circle
    cx = (patch_center_x - patch_x) * scale
    cy = (patch_center_y - patch_y) * scale
    r = spot_radius * scale
    circle = plt.Circle((cx, cy), r, fill=False, edgecolor='yellow', linewidth=1.5, linestyle='--')
    ax.add_patch(circle)

    # Build lookup: cell_id -> is_in
    in_set = set()
    for tag in cell_ids_tagged:
        if tag.endswith('_in'):
            in_set.add(tag[:-3])

    # Draw cell segmentation polygons
    for cell_id in cell_ids:
        if cell_id not in cell_dict:
            continue
        cell_shape = cell_dict[cell_id]['cell_shape']
        is_in = cell_id in in_set
        color = 'lime' if is_in else 'tomato'
        try:
            coords = np.array(cell_shape.exterior.coords)
        except AttributeError:
            continue  # skip MultiPolygon or degenerate
        coords_img = np.column_stack([
            (coords[:, 0] - patch_x) * scale,
            (coords[:, 1] - patch_y) * scale,
        ])
        poly_patch = plt.Polygon(coords_img, fill=False, edgecolor=color, linewidth=0.8, alpha=0.9)
        ax.add_patch(poly_patch)

    ax.set_xlim(0, target_input_px)
    ax.set_ylim(target_input_px, 0)  # image coords: y increases downward
    ax.set_title(f"{cells_in_spot} cells in spot | {sample_id}", fontsize=8)
    ax.axis('off')

    in_p  = mpatches.Patch(facecolor='none', edgecolor='lime',   label='In spot')
    out_p = mpatches.Patch(facecolor='none', edgecolor='tomato', label='Out of spot')
    ax.legend(handles=[in_p, out_p], loc='upper right', fontsize=6, framealpha=0.6)

    out_path = Path(debug_dir) / f"n{cells_in_spot:02d}_{patch_barcode}.png"
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def process_sample(sample_id, st, model, preprocess, device, batch_size, config,
                   visualize_cells_per_spot=False, vis_debug_dir=None):
    data_folder = config['data']['data_folder']
    dataset_variant = config['data']['dataset_variant']
    model_name = config['data']['model_name']

    print(f"Processing Visium sample {sample_id}")
    if os.path.exists(f'{data_folder}/expressions{dataset_variant}/{sample_id}_expressions.h5'):
        print(f"  Already processed {sample_id}, skipping")
        return
    
    print_system_status(f"Starting Sample {sample_id}")

    exp_name = config['experiment']['name']
    base_output_dir = config['experiment']['output_dir']
    stats_dir = Path(base_output_dir) / exp_name / "preprocessing_stats"
    os.makedirs(stats_dir, exist_ok=True)
    stats_file_path = stats_dir / f"{sample_id}_stats.json"

    initial_spot_count = st.adata.n_obs

    # Initialize cell distribution tracker
    # Indices 0-49 track exact counts, index 50 tracks "50 or more" cells
    cells_per_spot_distribution = [0] * 51

    # --- 1. Load Metadata and Config ---
    with open(f"{data_folder}/metadata/{sample_id}.json") as f:
        metadata = json.load(f)

    # Get resolution (microns per pixel)
    um_per_px = metadata.get('pixel_size_um_estimated', 0.5)

    # --- MICRON-BASED CALCULATIONS ---
    target_input_px = config['data'].get('target_input_size_px', 224) # Model Input (e.g. 224)
    target_context_um = config['data'].get('target_context_um', 112.0) # Physical Context (e.g. 112um)
    spot_diameter_um = config['data'].get('spot_diameter_um', 55.0)    # Physical Spot (e.g. 55um)

    # 1. How many pixels do we need to crop to get 112um?
    crop_size_px = int(target_context_um / um_per_px)
    
    # 2. How many pixels is the spot diameter on this specific slide?
    spot_diameter_px = int(spot_diameter_um / um_per_px)

    print(f"  Resolution Normalization:")
    print(f"    - Native Resolution: {um_per_px:.4f} um/px")
    print(f"    - Target Context:    {target_context_um} um -> Cropping {crop_size_px} px")
    print(f"    - Target Spot Dia:   {spot_diameter_um} um -> {spot_diameter_px} px")
    print(f"    - Model Input:       {target_input_px} px (Downsampling factor: {crop_size_px/target_input_px:.2f}x)")

    # For neighbor logic (if used elsewhere), we consider the 'patch_size' to be the crop size
    patch_size = crop_size_px

    print("  Standardizing gene names...")
    st.adata = standardize_gene_names(st.adata)

    # --- Spot-level QC parameters ---
    min_spot_counts    = config['data'].get('min_spot_counts', 100)
    min_spot_genes     = config['data'].get('min_spot_genes',  50)
    max_mt_fraction    = config['data'].get('max_mt_fraction', 0.30)   # 30% = clearly apoptotic/dying spots

    # --- Mitochondrial fraction (computed BEFORE gene subsetting; MT genes are still in adata here) ---
    # Note: MT- and MTRNR genes are excluded from the master gene list via is_artifact(),
    # so this fraction is used only as a QC signal, not as a training target.
    mt_genes_mask = np.array([g.startswith('MT-') or g.startswith('MTRNR')
                               for g in st.adata.var_names])
    n_mt_genes = int(mt_genes_mask.sum())
    if n_mt_genes > 0:
        X_dense = st.adata.X.toarray() if hasattr(st.adata.X, 'toarray') else np.array(st.adata.X)
        total_counts_vec = X_dense.sum(axis=1)
        mt_counts_vec    = X_dense[:, mt_genes_mask].sum(axis=1)
        del X_dense
        mt_fraction_vec  = np.where(total_counts_vec > 0,
                                    mt_counts_vec / total_counts_vec, 0.0)
        st.adata.obs['mt_fraction'] = mt_fraction_vec
        print(f"  MT genes in sample: {n_mt_genes} | per-spot MT fraction: "
              f"mean={mt_fraction_vec.mean():.3f}, median={np.median(mt_fraction_vec):.3f}, "
              f"max={mt_fraction_vec.max():.3f}")
    else:
        mt_fraction_vec = np.zeros(st.adata.n_obs)
        st.adata.obs['mt_fraction'] = mt_fraction_vec
        print("  No MT- genes found in this sample. MT fraction filter will be skipped.")

    # --- Spot filter 1: minimum total counts ---
    count_pre = st.adata.n_obs
    sc.pp.filter_cells(st.adata, min_counts=min_spot_counts)
    spots_low_count_filtered = count_pre - st.adata.n_obs
    print(f"  Spot filter min_counts >= {min_spot_counts}: "
          f"{count_pre} -> {st.adata.n_obs} (removed {spots_low_count_filtered})")

    # --- Spot filter 2: minimum unique genes ---
    gene_pre = st.adata.n_obs
    sc.pp.filter_cells(st.adata, min_genes=min_spot_genes)
    spots_low_genes_filtered = gene_pre - st.adata.n_obs
    print(f"  Spot filter min_genes >= {min_spot_genes}: "
          f"{gene_pre} -> {st.adata.n_obs} (removed {spots_low_genes_filtered})")

    # --- Spot filter 3: mitochondrial fraction (optional, very lenient) ---
    if max_mt_fraction is not None and n_mt_genes > 0:
        mt_pre = st.adata.n_obs
        st.adata = st.adata[st.adata.obs['mt_fraction'] <= max_mt_fraction].copy()
        spots_mt_filtered = mt_pre - st.adata.n_obs
        print(f"  Spot filter max_mt_fraction <= {max_mt_fraction:.0%}: "
              f"{mt_pre} -> {st.adata.n_obs} (removed {spots_mt_filtered})")
    else:
        spots_mt_filtered = 0

    print(f"  Spots remaining after all QC filters: {st.adata.n_obs} "
          f"(of {initial_spot_count} initial)")

    ordered_gene_names, gene_to_ordered_idx = load_master_gene_list(st.adata.var.index, config)

    master_set = set(ordered_gene_names)
    sample_set = set(st.adata.var.index)
    common = master_set.intersection(sample_set)
    overlap_pct = (len(common) / len(master_set)) * 100 if len(master_set) > 0 else 0.0

    num_master_genes = len(ordered_gene_names)

    # Load tissue mask
    with open(f"{data_folder}/tissue_seg/{sample_id}_contours.geojson", 'r') as f:
        tissue_mask_json = json.load(f)
    tissue_mask = create_tissue_mask(tissue_mask_json, sample_id)
    if tissue_mask is None:
        print(f"  FATAL: Could not create tissue mask for {sample_id}. Skipping.")
        return

    # Load WSI
    print_system_status("Loading WSI to RAM")
    wsi_path = f"{data_folder}/wsis/{sample_id}.tif"
    if not os.path.exists(wsi_path):
        print(f"  FATAL: WSI not found at {wsi_path}. Skipping.")
        return
    wsi = pyvips.Image.new_from_file(wsi_path)

    # --- 2. Load Cell Segmentations ---
    # Default to True if the flag isn't in the config file
    use_cellvit = config['data'].get('use_cellvit', True)
    
    cellvit_plus_path = f"{data_folder}/cellvit_plus_seg/{sample_id}.geojson"

    if use_cellvit and os.path.exists(cellvit_plus_path):
        # Load custom CellViT++ segmentation
        print(f"  Loading custom CellViT++ segmentation from: {cellvit_plus_path}")
        
        # --- BYPASS GDAL FOR NON-STANDARD GEOJSON ARRAYS ---
        print(f"  Reading raw JSON array into memory...")
        with open(cellvit_plus_path, 'r') as f:
            raw_features = json.load(f)
            
        print(f"  Converting {len(raw_features)} class-grouped features to GeoDataFrame...")
        cell_shapes = gpd.GeoDataFrame.from_features(raw_features)
        
        # --- EXPLODE MULTIPOLYGONS ---
        # Shatter the 5 giant class groupings into hundreds of thousands of individual cell polygons
        cell_shapes = cell_shapes.explode(ignore_index=True)
        
        # --- GENERATE UNIQUE CELL IDs ---
        # Assign fresh, unique string IDs so the KDTree and dictionaries don't overwrite
        cell_shapes.index = [f"cellvit_{i}" for i in range(len(cell_shapes))]
        
        print(f"  Successfully extracted {len(cell_shapes)} individual cells.")
        # ---------------------------------------------------

    else:
        # Fallback to HEST's built-in segmentation
        if use_cellvit:
            print(f"  Warning: CellViT requested but not found at {cellvit_plus_path}. Falling back to HEST built-in.")
        else:
            print(f"  CellViT disabled in config. Using HEST built-in segmentation.")
            
        target_shape_obj = None

        # Robustly find the segmentation object
        if hasattr(st, 'shapes'):
            for s in st.shapes:
                name = getattr(s, 'name', str(s))
                if 'cellvit' in name.lower():
                    target_shape_obj = s
                    break

            # Fallback to the first available segmentation if 'cellvit' not found
            if target_shape_obj is None and len(st.shapes) > 0:
                target_shape_obj = st.shapes[0]
                print("  cellvit in shape not found! ")

        if target_shape_obj is None:
            print(f"  FATAL: No cell segmentation found for {sample_id}.")
            return

        print(f"  Using segmentation: {getattr(target_shape_obj, 'name', 'unknown')}")

        # Extract shapes
        if hasattr(target_shape_obj, 'shapes'):
            cell_shapes = target_shape_obj.shapes
        else:
            cell_shapes = st.get_shapes(target_shape_obj, 'he').shapes

        # Decode index if stored as bytes (only needed for HEST fallback)
        if cell_shapes.index.dtype == 'object':
            decode = lambda x: x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x)
            cell_shapes.index = cell_shapes.index.map(decode).astype("string")

        print(f"  Found {len(cell_shapes)} cell segmentations.")

    # --- 3. Pre-process Cell Embeddings (MICRON AWARE) ---
    min_cell_context_um = config['data'].get('min_cell_context_um', 36.0)
    min_cell_crop_px = int(min_cell_context_um / um_per_px)
    print(f"  Dynamic Min Cell Crop: {min_cell_crop_px} px ({min_cell_context_um} um)")

    # --- Cell area QC parameters (morphological filter on CellViT polygons) ---
    min_cell_area_um2 = config['data'].get('min_cell_area_um2', 6.0)
    max_cell_area_um2 = config['data'].get('max_cell_area_um2', 400.0)

    cell_dict = {}
    cell_batch = []
    cell_batch_ids = []
    total_cells = 0
    cells_inside_tissue      = 0
    cells_filtered_area_small = 0
    cells_filtered_area_large = 0
    cell_start_time = time.time()

    print(f"  Starting processing of {len(cell_shapes)} cell images...")
    for cell_id, cell_row in cell_shapes.iterrows():
        total_cells += 1
        if total_cells % 10000 == 0:
            elapsed = time.time() - cell_start_time
            cells_per_sec = total_cells / elapsed
            print(f"  Processed {total_cells}/{len(cell_shapes)} cell images ({cells_per_sec:.0f} cells/sec)")
            print_system_status(f"Cell Processing Loop {total_cells}")

        cell_shape = cell_row.geometry
        if not cell_shape.is_valid:
            cell_shape = fix_invalid_geometry(cell_shape)
            if not cell_shape.is_valid:
                continue

        if cell_shape.is_empty:
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

        # Area bounds filter: removes debris fragments (too small) and merged-cell artifacts (too large)
        cell_area = cell_shape.area
        cell_area_um2 = cell_area * (um_per_px ** 2)
        if cell_area_um2 < min_cell_area_um2:
            cells_filtered_area_small += 1
            continue
        if max_cell_area_um2 is not None and cell_area_um2 > max_cell_area_um2:
            cells_filtered_area_large += 1
            continue

        cell_bounds = cell_shape.bounds  # (minx, miny, maxx, maxy)

        # Calculate dimensions
        rect_width = cell_bounds[2] - cell_bounds[0]
        rect_height = cell_bounds[3] - cell_bounds[1]
        
        # Enforce minimum crop size
        crop_size = max(rect_width, rect_height, min_cell_crop_px)
        max_dimension = int(np.ceil(crop_size))

        # Safely convert to integer pixel coordinates
        square_minx = int(round(cell_centroid_x - max_dimension / 2.0))
        square_miny = int(round(cell_centroid_y - max_dimension / 2.0))

        try:
            cell_img = crop_tile(wsi, square_minx, square_miny, max_dimension)
        except Exception as e:
            # print(f"  Warning: Error cropping cell {cell_id} image: {e}")
            continue

        # --- EXPLICIT RESIZE FOR CELLS ---
        # Resize cell crop to 224x224 so the batch stack works and model sees normalized input
        cell_img_pil = Image.fromarray(cell_img)
        if cell_img_pil.size != (target_input_px, target_input_px):
            cell_img_pil = cell_img_pil.resize((target_input_px, target_input_px), resample=Image.LANCZOS)
        cell_img = np.array(cell_img_pil)

        # Store data needed for batch processing
        cell_batch.append(cell_img)
        cell_batch_ids.append(cell_id)

        # Store data needed for spot-cell mapping
        cell_dict[cell_id] = {
            'cell_shape': cell_shape,
            'cell_centroid_x': cell_centroid_x,
            'cell_centroid_y': cell_centroid_y,
            'area': cell_area,  # already computed above for area filter
            'cell_bounds': cell_bounds,
            'square_bbox_coords': (square_minx, square_miny, square_minx + max_dimension, square_miny + max_dimension),
            'cell_embedding': None # Will be filled in batch
        }

        # Process batch when full
        if len(cell_batch) == batch_size:
            cell_embeddings = process_batch(cell_batch, model, preprocess, device)
            for j, c_id in enumerate(cell_batch_ids):
                if c_id in cell_dict:
                    cell_dict[c_id]['cell_embedding'] = cell_embeddings[j]
            cell_batch = []
            cell_batch_ids = []

    # Process final remaining batch
    if len(cell_batch) > 0:
        cell_embeddings = process_batch(cell_batch, model, preprocess, device)
        for j, c_id in enumerate(cell_batch_ids):
             if c_id in cell_dict:
                cell_dict[c_id]['cell_embedding'] = cell_embeddings[j]

    elapsed = time.time() - cell_start_time
    print(f"  Completed cell processing: {cells_inside_tissue}/{total_cells} cells inside tissue in {elapsed:.1f} sec")
    print(f"  Cell area QC: removed {cells_filtered_area_small} too-small (<{min_cell_area_um2} μm²), "
          f"{cells_filtered_area_large} too-large (>{max_cell_area_um2} μm²)")

    # Filter cell_dict to only include cells that were successfully processed
    processed_cell_ids = {cid for cid, data in cell_dict.items() if data['cell_embedding'] is not None}
    cell_dict = {cid: data for cid, data in cell_dict.items() if cid in processed_cell_ids}
    print(f"  {len(cell_dict)} cells successfully processed with embeddings.")

    print("  Building cKDTree for cell spatial queries...")
    valid_cell_ids_list = list(cell_dict.keys())
    cell_centroids = np.array([
        [cell_dict[cid]['cell_centroid_x'], cell_dict[cid]['cell_centroid_y']] 
        for cid in valid_cell_ids_list
    ])
    cell_kdtree = cKDTree(cell_centroids)


    # --- 4. Iterate over REAL Visium Spots ---
    patches_inside_tissue = 0
    patches_outside_tissue = 0
    patches_visual_filtered = 0
    patches_no_cells = 0
    patches_no_inner_cells = 0
    patches_saved = 0
    unique_saved_cells = set()
    total_cell_patch_assignments = 0
    cells_per_spot_vis_count = defaultdict(int)  # tracks how many debug images saved per count

    h5_output_path = f'{data_folder}/expressions{dataset_variant}/{sample_id}_expressions.h5'
    h5_file = h5py.File(h5_output_path, 'w')

    patch_start_time = time.time()
    total_spots_to_process = len(st.adata)
    print(f"  Starting processing of {total_spots_to_process} Visium spots...")
    for spot_count, (original_barcode, spot_data) in enumerate(st.adata.obs.iterrows()):
        if spot_count % 1000 == 0 and spot_count > 0:
            elapsed = time.time() - patch_start_time
            spots_per_sec = spot_count / elapsed
            remaining = (total_spots_to_process - spot_count) / spots_per_sec if spots_per_sec > 0 else 0
            print(f"  Processed {spot_count}/{total_spots_to_process} spots ({spot_count/total_spots_to_process*100:.1f}%) - "
                f"ETA: {int(remaining/60)} min {int(remaining)%60} sec")
            print_system_status(f"Spot Processing Loop {spot_count}")

        # 1. Extract Array Coordinates (Row/Col)
        if 'array_row' in spot_data and 'array_col' in spot_data:
            # This will execute for Visium
            row = int(spot_data['array_row'])
            col = int(spot_data['array_col'])
        else:
            # CRITICAL WARNING for Visium
            print(f"WARNING: Spot {original_barcode} missing array coordinates. Fallback to square grid may break neighbor graph.")
            row = int(spot_data['pxl_row_in_fullres'] // patch_size)
            col = int(spot_data['pxl_col_in_fullres'] // patch_size)

        # 2. Create the Xenium-style Filename
        patch_barcode = f"patch_{row}_{col}_{sample_id}"

        # Get spot coordinates
        patch_center_x = spot_data['pxl_col_in_fullres']
        patch_center_y = spot_data['pxl_row_in_fullres']
        patch_x = patch_center_x - patch_size // 2
        patch_y = patch_center_y - patch_size // 2

        # Skip spots whose centroid is outside tissue
        if not tissue_mask.contains(Point(patch_center_x, patch_center_y)):
            patches_outside_tissue += 1
            continue

        patches_inside_tissue += 1

        # Find nearby cells from our pre-processed cell_dict
        search_radius = patch_size 
        
        # Use p=np.inf to create a square search bounding box exactly like your old math
        nearby_idx = cell_kdtree.query_ball_point([patch_center_x, patch_center_y], r=search_radius, p=np.inf)
        nearby_cells = {valid_cell_ids_list[i]: cell_dict[valid_cell_ids_list[i]] for i in nearby_idx}

        if len(nearby_cells) == 0:
            patches_no_cells += 1
            continue

        # Crop the main patch H&E image
        try:
            patch_img = crop_tile(wsi, patch_x, patch_y, patch_size)
        except Exception:
            continue

        # --- EXPLICIT RESIZE FOR PATCHES ---
        # Normalize to 224x224 (Target Input) using LANCZOS
        patch_img_pil = Image.fromarray(patch_img)
        if patch_img_pil.size != (target_input_px, target_input_px):
            patch_img_pil = patch_img_pil.resize((target_input_px, target_input_px), resample=Image.LANCZOS)
        
        # Convert back to array for processing
        patch_img = np.array(patch_img_pil)

        if not check_patch_quality(patch_img):
            patches_visual_filtered += 1
            continue

        patch_box = box(patch_x, patch_y, patch_x + patch_size, patch_y + patch_size)
        cell_bboxes_adjusted = []
        cell_centroids_adjusted = []
        cell_ids = []
        cell_ids_tagged = []
        cell_embeddings_for_patch = [] # To store embeddings for .npy file

        if not patch_box.is_valid:
            patch_box = fix_invalid_geometry(patch_box)
            if not patch_box.is_valid:
                continue # Skip invalid patch geometry

        spot_radius = spot_diameter_px // 2
        spot_geom = Point(patch_center_x, patch_center_y).buffer(spot_radius, cap_style=1)

        for cell_id, cell_data in nearby_cells.items():
            cell_shape = cell_data['cell_shape']
            cell_bounds = cell_data['cell_bounds']
            patch_bounds = patch_box.bounds  # (minx, miny, maxx, maxy)

            # Simple bounding box check first for speed
            if (cell_bounds[2] < patch_bounds[0] or cell_bounds[0] > patch_bounds[2] or
                cell_bounds[3] < patch_bounds[1] or cell_bounds[1] > patch_bounds[3]):
                continue

            # Check for sufficient intersection with the patch
            try:
                if cell_shape.intersects(patch_box):
                    intersection = cell_shape.intersection(patch_box)
                    if intersection.area / cell_data['area'] < 0.6:
                        continue
                else:
                    continue
            except Exception:
                continue # Skip if intersection check fails

            # Check if cell centroid is inside the spot
            if spot_geom.contains(Point(cell_data['cell_centroid_x'], cell_data['cell_centroid_y'])):
                cell_id_tagged = f"{cell_id}_in"
            else:
                cell_id_tagged = f"{cell_id}_out"

            # --- Add cell to this patch's data ---
            cell_ids.append(cell_id)
            cell_ids_tagged.append(cell_id_tagged)
            cell_embeddings_for_patch.append(cell_data['cell_embedding'])

            # Adjust and normalize coordinates relative to the patch
            cell_bbox_coords = cell_data['square_bbox_coords']
            cell_bbox_adjusted_normalized = (
                (cell_bbox_coords[0] - patch_x) / patch_size,  # normalized minx
                (cell_bbox_coords[1] - patch_y) / patch_size,  # normalized miny
                (cell_bbox_coords[2] - patch_x) / patch_size,  # normalized maxx
                (cell_bbox_coords[3] - patch_y) / patch_size   # normalized maxy
            )
            cell_bboxes_adjusted.append(cell_bbox_adjusted_normalized)
            cell_centroid_x = (cell_data['cell_centroid_x'] - patch_x) / patch_size
            cell_centroid_y = (cell_data['cell_centroid_y'] - patch_y) / patch_size
            cell_centroids_adjusted.append((cell_centroid_x, cell_centroid_y))

        if len(cell_ids) == 0:
            patches_no_cells += 1
            continue

        # --- 5. Get TRUE Spot Expression ---
        try:
            spot_adata = st.adata[original_barcode]
            spot_genes_local = spot_adata.var.index.tolist()
            if hasattr(spot_adata.X, "toarray"):
                spot_expr_raw = spot_adata.X.toarray().squeeze()
            else:
                spot_expr_raw = np.array(spot_adata.X).squeeze()

            # Create a new vector in the master gene order
            spot_expr_ordered = np.zeros(num_master_genes, dtype=np.float32)

            # Map genes from local order to master order
            for i, gene in enumerate(spot_genes_local):
                if gene in gene_to_ordered_idx:
                    ordered_idx = gene_to_ordered_idx[gene]
                    spot_expr_ordered[ordered_idx] = spot_expr_raw[i]

        except Exception as e:
            print(f"  Warning: Could not get expression for spot {original_barcode}. Error: {e}")
            continue

        # Only count cells that were tagged with '_in' (meaning they intersect the 55um spot)
        cells_in_spot = sum(1 for tag in cell_ids_tagged if tag.endswith('_in'))

        # Skip spots with no inner cells — model has nothing cell-specific to learn from
        if cells_in_spot == 0:
            patches_no_inner_cells += 1
            continue

        if cells_in_spot < 50:
            cells_per_spot_distribution[cells_in_spot] += 1
        else:
            cells_per_spot_distribution[50] += 1

        # --- 5b. Debug Visualization (optional) ---
        if visualize_cells_per_spot and cells_per_spot_vis_count[cells_in_spot] < 3:
            save_spot_visualization(
                patch_img=patch_img,
                patch_x=patch_x, patch_y=patch_y,
                patch_size=patch_size, target_input_px=target_input_px,
                spot_radius=spot_radius,
                patch_center_x=patch_center_x, patch_center_y=patch_center_y,
                cell_ids=cell_ids, cell_ids_tagged=cell_ids_tagged,
                cell_dict=cell_dict,
                cells_in_spot=cells_in_spot,
                sample_id=sample_id, patch_barcode=patch_barcode,
                debug_dir=vis_debug_dir,
            )
            cells_per_spot_vis_count[cells_in_spot] += 1

        # --- 6. Save Data for this Spot/Patch (incremental HDF5 write) ---
        dummy_cell_expressions = np.zeros((len(cell_ids), num_master_genes), dtype=np.float32)

        spot_group = h5_file.create_group(patch_barcode)
        spot_group.attrs['spot_diameter'] = spot_diameter_px
        spot_group.attrs['patch_size'] = patch_size
        spot_group.attrs['um_per_px'] = um_per_px
        spot_group.create_dataset('spot_expression', data=spot_expr_ordered, compression='gzip', compression_opts=4)
        spot_group.create_dataset('cell_expressions', data=dummy_cell_expressions, compression='gzip', compression_opts=4)
        cell_ids_array = np.array(cell_ids_tagged, dtype=h5py.string_dtype())
        spot_group.create_dataset('cell_ids', data=cell_ids_array)
        spot_group.create_dataset('cell_centroids', data=np.array(cell_centroids_adjusted, dtype=np.float32), compression='gzip', compression_opts=4)
        spot_group.create_dataset('cell_bbxs', data=np.array(cell_bboxes_adjusted, dtype=np.float32), compression='gzip', compression_opts=4)
        patches_saved += 1
        if patches_saved % 500 == 0:
            h5_file.flush()

        unique_saved_cells.update(cell_ids)
        total_cell_patch_assignments += len(cell_ids)

        # Get spot (patch) embedding
        spot_embedding = get_spot_embs(patch_img, model, preprocess, device)

        # Save patch and cell embeddings to .npy
        patch_embeddings = [spot_embedding.astype(np.float16)]
        for cell_emb in cell_embeddings_for_patch:
            if torch.is_tensor(cell_emb):
                cell_emb = cell_emb.numpy()
            patch_embeddings.append(cell_emb.astype(np.float16))

        patch_embeddings_stack = np.vstack(patch_embeddings)
        
        np.save(f'{data_folder}/embeddings{dataset_variant}/{model_name}/{patch_barcode}.npy', patch_embeddings_stack)

    h5_file.close()

    elapsed = time.time() - patch_start_time
    total_checked = total_spots_to_process

    print(f"--- Processing Summary for {sample_id} ---")
    print(f"  Time Elapsed:          {elapsed:.1f} sec")
    print(f"  Initial Spots:         {initial_spot_count}")
    print(f"  Low Count Filtered:    {spots_low_count_filtered}")
    print(f"  Spots outside Tissue:  {patches_outside_tissue}")
    print(f"  Visual QC Filtered:    {patches_visual_filtered}")
    print(f"  No Cells nearby:       {patches_no_cells}")
    print(f"  No Inner Cells (0 _in):{patches_no_inner_cells}")
    print(f"  TOTAL SAVED PATCHES:   {patches_saved}")
    print(f"  TOTAL UNIQUE CELLS:    {len(unique_saved_cells)}")
    print(f"------------------------------------------")
    print(f"  Saved spot data incrementally to {h5_output_path}")
    print(f"  Processed {patches_inside_tissue} spots inside tissue.")

    stats_data = {
        "sample_id": sample_id,
        "resolution_um_px": float(um_per_px),
        "gene_overlap_pct": round(overlap_pct, 2),
        "initial_spots": int(initial_spot_count),
        "spots_outside_tissue": int(patches_outside_tissue),
        "visual_qc_filtered": int(patches_visual_filtered),
        "no_cells_nearby": int(patches_no_cells),
        "no_inner_cells_filtered": int(patches_no_inner_cells),
        "final_saved_patches": int(patches_saved),

        "total_cells_segmented": int(total_cells),
        "cells_inside_tissue": int(cells_inside_tissue),
        "valid_cells_embedded": int(len(cell_dict)),
        "unique_cells_in_saved_patches": int(len(unique_saved_cells)),
        "total_cell_patch_assignments": int(total_cell_patch_assignments),

        # --- Spot QC filter statistics ---
        "qc_thresholds": {
            "min_spot_counts":   min_spot_counts,
            "min_spot_genes":    min_spot_genes,
            "max_mt_fraction":   max_mt_fraction,
            "min_cell_area_um2": min_cell_area_um2,
            "max_cell_area_um2": max_cell_area_um2,
        },
        "qc_spots_pre_filter":         int(initial_spot_count),
        "qc_spots_post_filter":        int(st.adata.n_obs),
        "qc_spots_min_counts_removed": int(spots_low_count_filtered),
        "qc_spots_min_genes_removed":  int(spots_low_genes_filtered),
        "qc_spots_mt_removed":         int(spots_mt_filtered),
        "qc_mt_genes_in_sample":       int(n_mt_genes),
        "qc_mt_fraction_mean":         round(float(mt_fraction_vec.mean()), 4),
        "qc_mt_fraction_median":       round(float(np.median(mt_fraction_vec)), 4),
        "qc_mt_fraction_max":          round(float(mt_fraction_vec.max()), 4),
        "qc_cells_area_small_removed": int(cells_filtered_area_small),
        "qc_cells_area_large_removed": int(cells_filtered_area_large),

        "is_xenium": 0.,
        "cells_per_spot_distribution": cells_per_spot_distribution
    }

    try:
        with open(stats_file_path, 'w') as f:
            json.dump(stats_data, f, indent=4)
        print(f"  Saved statistics to {stats_file_path}")
    except Exception as e:
        print(f"  WARNING: Failed to save statistics json: {e}")

    del cell_dict, st, wsi
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Visium slides')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config file')
    parser.add_argument('--split', type=str, required=True, help='Split name to process (e.g., train, val)')
    parser.add_argument('--indices', type=int, nargs='+', help='List of specific indices to process')
    parser.add_argument('--debug', action='store_true', help='Enable performance profiling')
    parser.add_argument('--visualize_cells_per_spot', action='store_true', default=False,
                        help='Save up to 3 example spot visualizations per cells-per-spot count for debugging')
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

    exp_name = config['experiment']['name']
    base_output_dir = config['experiment']['output_dir']

    for i, st in enumerate(iter_hest(data_folder, id_list=ids_to_process, load_transcripts=False)):
        sample_id = ids_to_process[i]

        vis_debug_dir = None
        if args.visualize_cells_per_spot:
            vis_debug_dir = Path(base_output_dir) / exp_name / "debug_cells_per_spot" / sample_id
            vis_debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Debug visualizations will be saved to: {vis_debug_dir}")

        process_sample(sample_id, st, model, preprocess, device, batch_size, config,
                       visualize_cells_per_spot=args.visualize_cells_per_spot,
                       vis_debug_dir=vis_debug_dir)