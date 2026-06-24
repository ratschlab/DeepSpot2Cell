import argparse
import os
import psutil
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from lightning.pytorch.loggers import CSVLogger
import torch
import yaml
import numpy as np

from deepspot2cell import DeepSpot2Cell, DS2CDataset
from deepspot2cell.utils.utils import fix_seed

# Global Debug Flag
DEBUG_MODE = False


def debug_print(*args, **kwargs):
    """Helper to print only if DEBUG_MODE is True."""
    if DEBUG_MODE:
        print(*args, **kwargs)


# --- DEBUG UTILITIES ---
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


class DebugMemoryCallback(Callback):
    """Custom callback to monitor VRAM during the training loop."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not DEBUG_MODE:
            return

        if batch_idx % 10 == 0:  # Don't spam every step, but frequent enough
            print_system_status(f"TRAIN STEP {batch_idx}")

            # Inspect Input Shapes for potential explosions
            cell_emb = batch["cell_embeddings"]
            spot_expr = batch["spot_expression"]
            # cell_emb shape: [Batch, Max_Cells, Dim]

            if batch_idx == 0:
                print(f"   -> Input Shape: {cell_emb.shape} (Type: {cell_emb.dtype})")
                print(
                    f"   -> Target Shape: {spot_expr.shape} (Type: {spot_expr.dtype})"
                )


def load_config(path: Path) -> dict:
    debug_print(f"DEBUG: Loading config from {path}")
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return cfg


def save_dataset_statistics(train_ds, output_dir: Path):
    """Saves normalization statistics for inference reuse."""
    stats_path = output_dir / "dataset_stats.pt"

    debug_print("DEBUG: Extracting dataset statistics...")
    # Add error checking
    if train_ds.standard_scaling and train_ds.spot_scaler is None:
        debug_print("[WARNING] standard_scaling is True but spot_scaler is None!")

    stats = {
        "standard_scaling": train_ds.standard_scaling,
        "minmax": train_ds.minmax,
        "normalize": train_ds.normalize,
        "norm_counts": train_ds.norm_counts,
        "spot_scaler": train_ds.spot_scaler,
        "cell_scaler": train_ds.cell_scaler,
        "spot_gene_min": train_ds.spot_gene_min,
        "spot_gene_max": train_ds.spot_gene_max,
        "spot_gene_range": train_ds.spot_gene_range,
        "cell_gene_min": train_ds.cell_gene_min,
        "cell_gene_max": train_ds.cell_gene_max,
        "cell_gene_range": train_ds.cell_gene_range,
        "num_genes": train_ds.num_genes,
    }
    torch.save(stats, stats_path)
    print(f"[OK] Saved dataset normalization statistics to: {stats_path}")


def build_datasets(cfg: dict):
    debug_print("DEBUG: Building Datasets...")
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    norm_cfg = cfg["normalization"]

    common_kwargs = dict(
        dataset_variant=data_cfg["dataset_variant"],
        data_path=data_cfg["data_folder"],
        model_name=data_cfg["model_name"],
        # --- Normalization ---
        standard_scaling=norm_cfg["standard_scaling"],
        normalize=norm_cfg["normalize"],
        minmax=norm_cfg["minmax"],
        norm_counts=norm_cfg["norm_counts"],
        # --- Feature Flags ---
        scellst=data_cfg.get("scellst", False),
        load_cell_types=data_cfg.get("load_cell_types", False),
        # --- Context ---
        neighb_degree=model_cfg.get("neighb_degree", 0),
        # --- Balancing ---
        max_patches_per_sample=data_cfg.get("max_patches_per_sample", None),
        target_patches_per_sample=data_cfg.get("target_patches_per_sample", None),
    )

    debug_print(f"DEBUG: Common Dataset Args: {common_kwargs}")

    # Read specific Train/Val splits
    train_split = data_cfg["split"]["train"]
    val_split = data_cfg["split"].get("val")

    debug_print(f"DEBUG: Train IDs: {len(train_split['ids'])} samples")

    if not train_split or not train_split.get("ids"):
        raise ValueError("No training IDs found in config['data']['split']['train']")

    print_system_status("Pre-TrainDS-Init")
    train_ds = DS2CDataset(
        ids_list=train_split["ids"],
        cell_gt_available=train_split.get("cell_gt_available", False),
        shuffle=True,
        **common_kwargs,
    )
    print_system_status("Post-TrainDS-Init")
    debug_print(f"DEBUG: Train DS Length: {len(train_ds)} patches")

    val_ds = None
    if val_split and val_split.get("ids"):
        debug_print(f"DEBUG: Val IDs: {len(val_split['ids'])} samples")
        val_ds = DS2CDataset(
            ids_list=val_split["ids"],
            cell_gt_available=val_split.get("cell_gt_available", False),
            shuffle=False,
            **common_kwargs,
        )

        # Share scaling statistics
        debug_print("DEBUG: Sharing scaler statistics with Val DS...")
        if train_ds.standard_scaling and train_ds.normalize:
            val_ds.cell_scaler = train_ds.cell_scaler
            val_ds.spot_scaler = train_ds.spot_scaler
        if train_ds.minmax:
            for attr in [
                "spot_gene_min",
                "spot_gene_max",
                "spot_gene_range",
                "cell_gene_min",
                "cell_gene_max",
                "cell_gene_range",
            ]:
                setattr(val_ds, attr, getattr(train_ds, attr))

    return train_ds, val_ds


def build_dataloaders(train_ds, val_ds, cfg: dict):
    dl_cfg = cfg.get("dataloader", {})
    batch_size = dl_cfg.get("batch_size", 4)
    train_num_workers = dl_cfg.get("num_workers", 0)
    val_num_workers = train_num_workers // 2 if train_num_workers > 2 else 0
    pin_memory = torch.cuda.is_available()

    debug_print(
        f"DEBUG: Building Dataloaders (BS={batch_size}, Train Workers={train_num_workers}, Val Workers={val_num_workers})"
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader


def build_model(train_ds, cfg: dict) -> DeepSpot2Cell:
    debug_print("DEBUG: Building Model...")

    sample = train_ds[0]
    input_size = sample["cell_embeddings"].shape[-1]
    output_size = sample["spot_expression"].shape[-1]

    debug_print(f"DEBUG: Model Input Size (Embedding Dim): {input_size}")
    debug_print(f"DEBUG: Model Output Size (Num Genes): {output_size}")

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    debug_print(f"DEBUG: Model Config: {model_cfg}")

    model = DeepSpot2Cell(
        input_size=input_size,
        output_size=output_size,
        lr=model_cfg.get("lr", 1e-4),
        weight_decay=model_cfg.get("weight_decay", 1e-6),
        p=model_cfg.get("dropout", 0.3),
        p_phi=model_cfg.get("dropout_phi"),
        p_rho=model_cfg.get("dropout_rho"),
        n_ensemble=model_cfg.get("n_ensemble", 10),
        n_ensemble_phi=model_cfg.get("n_ensemble_phi"),
        n_ensemble_rho=model_cfg.get("n_ensemble_rho"),
        phi2rho_size=model_cfg.get("phi2rho_size", 512),
        cell_gt_available=model_cfg.get("cell_gt_available", False),
        random_seed=cfg["experiment"]["random_seed"],
        use_attention=model_cfg.get("use_attention", False),
        loss_type=model_cfg.get("loss_type", "mse"),
        huber_delta=model_cfg.get("huber_delta", 1.0),
        context_dropout=model_cfg.get("context_dropout", 0.0),
        use_positional_encoding=model_cfg.get("use_positional_encoding", False),
        aux_loss_weight=model_cfg.get("aux_loss_weight", 0.0),
        pooling=model_cfg.get("pooling", "sum"),
        pool_layer_norm=model_cfg.get("pool_layer_norm", False),
        predict_then_sum=model_cfg.get("predict_then_sum", False),
        input_layer_norm=model_cfg.get("input_layer_norm", False),
        warmup_epochs=model_cfg.get("warmup_epochs", 0),
        cosine_epochs=model_cfg.get(
            "cosine_epochs", cfg["trainer"].get("max_epochs", 50)
        ),
        eta_min_factor=model_cfg.get("eta_min_factor", 0.01),
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    debug_print(f"DEBUG: Total Trainable Parameters: {params / 1e6:.2f} M")

    return model


def build_trainer(cfg: dict, output_dir: Path) -> L.Trainer:
    debug_print("DEBUG: Building Trainer...")
    trainer_cfg = cfg["trainer"]
    callbacks = []

    # Add our Memory Debugger
    callbacks.append(DebugMemoryCallback())

    early_cfg = trainer_cfg.get("early_stopping", {})
    if early_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=early_cfg.get("monitor", "val_loss"),
                patience=early_cfg.get("patience", 5),
                mode=early_cfg.get("mode", "min"),
                min_delta=early_cfg.get("min_delta", 0.0),
            )
        )

    ckpt_dir = output_dir / "checkpoints"
    # 1. Save the BEST model (Monitors val_loss)
    callbacks.append(
        ModelCheckpoint(
            dirpath=ckpt_dir,
            save_top_k=1,
            monitor=early_cfg.get("monitor", "val_loss"),
            mode=early_cfg.get("mode", "min"),
            filename="best-val",
            auto_insert_metric_name=False,
        )
    )

    # 2. Save the LAST model (No monitor, just saves the latest epoch)
    callbacks.append(
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="last",
            save_top_k=1,
            monitor=None,
        )
    )

    logger = CSVLogger(save_dir=output_dir, name="logs")

    precision = trainer_cfg.get("precision", 32)
    accum = trainer_cfg.get("accumulate_grad_batches", 1)

    debug_print(f"DEBUG: Trainer Precision: {precision}")
    debug_print(f"DEBUG: Gradient Accumulation: {accum}")

    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    if num_nodes > 1:
        debug_print(f"DEBUG: Multi-node training detected. Using {num_nodes} nodes.")

    return L.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 50),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        num_nodes=num_nodes,
        precision=precision,
        accumulate_grad_batches=accum,
        gradient_clip_val=trainer_cfg.get("gradient_clip_val"),
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
    )


def main():
    parser = argparse.ArgumentParser(description="Train DeepSpot2Cell")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    args = parser.parse_args()

    # optional for 4090
    torch.set_float32_matmul_precision("medium")

    global DEBUG_MODE
    DEBUG_MODE = args.debug

    print_system_status("Startup")

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    experiment_name = cfg["experiment"]["name"]
    output_dir = Path(cfg["experiment"]["output_dir"]).resolve() / experiment_name
    os.makedirs(output_dir, exist_ok=True)

    fix_seed(cfg["experiment"]["random_seed"])

    train_ds, val_ds = build_datasets(cfg)

    # Save stats
    save_dataset_statistics(train_ds, output_dir)

    train_loader, val_loader = build_dataloaders(train_ds, val_ds, cfg)

    # --- DATALOADER SANITY CHECK ---
    if DEBUG_MODE:
        print("\n--- DATALOADER SANITY CHECK ---")
        try:
            first_batch = next(iter(train_loader))
            print(f"Batch Keys: {first_batch.keys()}")
            print(f"Cell Emb Shape: {first_batch['cell_embeddings'].shape}")
            print(f"Ctx Emb Shape:  {first_batch['neighb_embeddings'].shape}")
            print(f"Spot Expr Shape: {first_batch['spot_expression'].shape}")
            print(f"Mask Shape:      {first_batch['cell_mask'].shape}")
            print("-------------------------------\n")
        except Exception as e:
            print(f"[FAILED] DATALOADER FAILED: {e}")
            import traceback

            traceback.print_exc()
            return
    # --------------------------------

    model = build_model(train_ds, cfg)
    print_system_status("Model Loaded")

    trainer = build_trainer(cfg, output_dir)

    debug_print("DEBUG: Starting Training Loop...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
