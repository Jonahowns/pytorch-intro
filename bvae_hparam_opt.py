import ray.tune as tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
import os
import numpy as np
import math
import argparse

# local files
from beta_VAE import BetaVAE


def tune_asha_search(config, num_samples=10, num_epochs=10, gpus_per_trial=0, cpus_per_trial=1):

    scheduler = tune.schedulers.ASHAScheduler(
        max_t=num_epochs,
        grace_period=math.floor(num_epochs/2),
        reduction_factor=2)

    reporter = tune.CLIReporter(
        parameter_columns=["lr", "batch_size", "beta"],
        metric_columns=["training_iteration", "train_loss", "train_recon_acc", "val_loss", "val_recon_acc"])

    # bayesopt = BayesOptSearch(metric="mean_loss", mode="min")
    analysis = tune.run(
        tune.with_parameters(
            train_bvae,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="val_loss",
        mode="min",
        local_dir="./ray_results/",
        config=config,
        num_samples=num_samples,
        # search_alg=bayesopt,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_bvae_asha",
        checkpoint_score_attr="val_loss",
        keep_checkpoints_num=1)

    print("Best hyperparameters found were: ", analysis.best_config)

def train_bvae(config, checkpoint_dir=None, num_epochs=10, num_gpus=0):
    trainer = Trainer(
        # default_root_dir="./checkpoints/",
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="tb", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={
                    "train_loss": "ptl/train_loss",
                    "val_loss": "ptl/val_loss",
                    "val_recon_acc": "ptl/val_recon_acc",
                    "train_recon_acc": "ptl/train_recon_acc"
                },
                filename="checkpoint",
                on="validation_end")
        ]
    )

    if checkpoint_dir:
        # Workaround:
        ckpt = pl_load(
            os.path.join(checkpoint_dir, "checkpoint"),
            map_location=lambda storage, loc: storage)
        model = BetaVAE._load_model_state(
            ckpt, config=config)
        trainer.current_epoch = ckpt["epoch"]
    else:
        model = BetaVAE(config)

    trainer.fit(model)

if __name__ == '__main__':
    os.environ["SLURM_JOB_NAME"] = "bash"   # server runs crash without this line (yay raytune)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of cuda errors

    # anything with tune.something in the config is something we are optimizing
    config = {"datatype": "HCB20",
              "in_channels": 1,
              "latent_dim": 50,
              "beta": tune.grid_search([1, 10, 100]),  # # only used if loss_type == "H", beta = 1 is standard VAE
              "gamma": 1,  # only used if loss_type == "B"
              "hidden_dims": [8, 16, 32, 64, 512],
              "max_capacity": 25,  # only used if loss_type == "B"
              "capacity_max_iter": 1e5,  # only used if loss_type == "B"
              "loss_type": "H",  # B or H
              "data_worker_num": 6,
              "optimizer": "AdamW",
              "lr": tune.loguniform(1e-5, 1e-2),
              "seed": np.random.randint(0, 100000, 1)[0],
              "batch_size": tune.grid_search([5000, 10000]),
              "replicas": 4,  # Number of samples for validation step
              "epochs": 100,
              "attention_layers": False,  # This doesn't quite work yet
              }

    # grid with 3 beta values and 2 batch_size values (6 points)
    # lr is initialized according to loguniform distribution but doesn't affect grid size
    # samples controls how many times each point on grid is sampled

    # raytune will automatically use all gpus available to it. So each gpu can run a trial in our case
    # I would recommend using 2-3 gpus to train this

    tune_asha_search(config, num_samples=2, num_epochs=100, gpus_per_trial=1, cpus_per_trial=4)
