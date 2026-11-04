import argparse
import datetime
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from project.config.model_config import RoadBEVModelConfig
from project.dataset.dataloader import ModelDataModule
from project.model.model import RoadBEVManager, RoadBEV

from tools.utils import (
    config_to_dict,
    import_module_from_file,
    save_config_files
)
from project.utils.wandb_visualization import WandbVisualizationCallback

import wandb
import yaml

# Safe globals registration for pickle serialization
torch.backends.cudnn.benchmark = False
try:
    from torch.serialization import add_safe_globals

    add_safe_globals([RoadBEVModelConfig])
except (ImportError, AttributeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (default: config/model_config.py)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training (restores model, optimizer, scheduler, epoch)",
    )
    
    return parser.parse_args()

def _init_wandb(config: RoadBEVModelConfig) -> Tuple[WandbLogger, str, str, str, Dict[str, Any], Dict[str, Any]]:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    applied_overrides: Dict[str, Any] = {}

    wandb_run_name = config.wandb_run_name
    run_id = f"{config.wandb_run_name}_{timestamp}"
    save_dir = Path("./exp/RoadBEV") / config.wandb_run_name
    logger = WandbLogger(
        entity=config.wandb_entity_name,
        project=config.wandb_project_name,
        name=wandb_run_name,
        id=run_id,
        log_model=False,
        settings=wandb.Settings(init_timeout=180),
        )

    logger.log_hyperparams(config_to_dict(config))

    runtime_metadata = {
        "wandb_run_name": wandb_run_name,
        "run_id": run_id,
        "save_dir": str(save_dir)
    }

    return logger, timestamp, run_id, str(save_dir), applied_overrides, runtime_metadata


def main():
    args = parse_args()

    module_path = Path(args.config).resolve()
    module = import_module_from_file("RoadBEVModelConfig", str(module_path))

    model_config = getattr(module, "RoadBEVModelConfig")
    allow = [model_config]
    try:
        add_safe_globals(allow)
    except (NameError, AttributeError):
        pass

    config = module.RoadBEVModelConfig()

    logger, timestamp, _, save_dir_str, _, _ = _init_wandb(config=config)
    save_dir = Path(save_dir_str)

    print("Using configuration from:", args.config)
    print("\n")
    print(config)

    pl.seed_everything(config.seed, workers=True)

    dm = ModelDataModule(config=config)
    model = RoadBEV(config=config)
    pl_module = RoadBEVManager(model=model, config=config)
    
    ckpt_cb = ModelCheckpoint(
        dirpath=str(save_dir / Path("checkpoints")),
        filename="{epoch:02d}",
        monitor="train_loss_epoch",
        mode="min",
        save_top_k=200,
        save_last=True,
        every_n_epochs=1,
        verbose=True,
    )

    lr_cb = LearningRateMonitor(logging_interval="epoch")
    vis_cb = WandbVisualizationCallback(config=config, num_samples=4)

    accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"accumulate grad batches: {config.accumulate_grad_batches}")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=config.devices,
        strategy=DDPStrategy(find_unused_parameters=False) if config.strategy == "ddp" else config.strategy,
        precision=config.precision,
        max_epochs=config.epochs,
        logger=logger,
        callbacks=[lr_cb, ckpt_cb, vis_cb],
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=5.0,
        gradient_clip_algorithm="norm",
    )

    save_config_files(save_dir=save_dir, timestamp=timestamp, config_file_path=args.config)

    trainer.fit(model=pl_module, datamodule=dm, ckpt_path=args.ckpt_path)

if __name__ == "__main__":
    main()
