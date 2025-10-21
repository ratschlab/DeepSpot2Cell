import argparse
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch
import yaml

from deepspot2cell import DeepSpot2Cell, DS2CDataset
from deepspot2cell.utils.utils import fix_seed


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_datasets(cfg: dict):
    data_cfg = cfg["data"]
    common_kwargs = dict(
        dataset_variant=data_cfg["dataset_variant"],
        data_path=data_cfg["data_folder"],
        model_name=data_cfg["model_name"],
        standard_scaling=data_cfg.get("standard_scaling", False),
        normalize=data_cfg.get("normalize", True),
        minmax=data_cfg.get("minmax", False),
        neighb_degree=data_cfg.get("neighb_degree", 0),
        norm_counts=data_cfg.get("norm_counts", 1e4),
        scellst=data_cfg.get("scellst", False),
        load_cell_types=data_cfg.get("load_cell_types", False),
    )

    train_ids = data_cfg.get("train_ids", [])
    val_ids = data_cfg.get("val_ids", [])

    if not train_ids:
        raise ValueError("No training sample ids provided in config.data.train_ids")

    train_ds = DS2CDataset(ids_list=train_ids, shuffle=True, **common_kwargs)
    val_ds = None
    if val_ids:
        val_ds = DS2CDataset(ids_list=val_ids, shuffle=False, **common_kwargs)
        # Share scaling statistics between splits
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
    dl_cfg = cfg["dataloader"]
    batch_size = dl_cfg.get("batch_size", 4)
    num_workers = dl_cfg.get("num_workers", 0)
    pin_memory = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader


def build_model(train_ds, cfg: dict) -> DeepSpot2Cell:
    sample = train_ds[0]
    input_size = sample["cell_embeddings"].shape[-1]
    output_size = sample["spot_expression"].shape[-1]

    model_cfg = cfg["model"]
    return DeepSpot2Cell(
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
        cell_gt_available=model_cfg.get("cell_gt_available", True),
        random_seed=cfg["experiment"]["random_seed"],
    )


def build_trainer(cfg: dict, output_dir: Path) -> L.Trainer:
    trainer_cfg = cfg["trainer"]
    callbacks = []

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
    callbacks.append(
        ModelCheckpoint(
            dirpath=ckpt_dir,
            save_top_k=1,
            monitor=early_cfg.get("monitor", "val_loss"),
            mode=early_cfg.get("mode", "min"),
            filename="best-{epoch}-{val_loss:.4f}",
            auto_insert_metric_name=False,
        )
    )

    logger = CSVLogger(save_dir=output_dir, name="logs")

    return L.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 50),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", 32),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val"),
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
    )


def main():
    parser = argparse.ArgumentParser(description="Train DeepSpot2Cell")
    parser.add_argument("--config", type=str, default="configs/training.example.yaml", help="Path to training config")
    parser.add_argument("--output", type=str, default="runs", help="Directory to store checkpoints and logs")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    experiment_name = cfg["experiment"]["name"]
    output_dir = Path(args.output).resolve() / experiment_name
    os.makedirs(output_dir, exist_ok=True)

    fix_seed(cfg["experiment"]["random_seed"])

    train_ds, val_ds = build_datasets(cfg)
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, cfg)
    model = build_model(train_ds, cfg)

    trainer = build_trainer(cfg, output_dir)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
