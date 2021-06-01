# run 10 motif split simulation script
import logging

import click
import numpy as np
from pathlib import Path
import optuna

import torch
from torch import optim

from src import models
from src.models import CNN_PHMM_VAE_FAST

from src.data import SequenceGenerator, SingleRound

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
default_path = str(Path(f"{dir_path}/../out/real").resolve())
counter = 0


@click.command(help='run experiment with real data', context_settings=dict(show_default=True))
@click.argument("seqpath", type=click.Path(exists=True))
@click.option("--epochs", help="the number of training epochs", type=int, default=1000)
@click.option("--threshold", help="the number of epochs with no loss update to stop training", type=int, default=50)
@click.option("--use-cuda/--no-cuda", help="use cuda if available", is_flag=True, default=True)
@click.option("--cuda-id", help="the device id of cuda to run", type=int, default=0)
@click.option("--save-dir", help="path to save results", type=click.Path(), default=default_path)
@click.option("--fwd", help="forward adapter", type=str, default=None)
@click.option("--rev", help="reverse adapter", type=str, default=None)
@click.option("--min-count", help="minimum duplication count to pass sequence for training", type=int, default=1)
@click.option("--multi", help="the number of training for multiple times", type=int, default=1)
@click.option("--reg-epochs", help="the number of epochs to conduct state transition regularization", type=int, default=50)
@click.option("--batch-size", help="the number of batch size", type=int, default=512)
@click.option("--storage", help="the file to save optimize results", type=click.Path(), default=None)
def main(seqpath, epochs, threshold, cuda_id, use_cuda,
         save_dir, fwd, rev, min_count, multi, reg_epochs, batch_size, storage):
    logger = logging.getLogger(__name__)

    logger.info(f"saving to {save_dir}")
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    experiment = SingleRound(
        path=seqpath,
        forward_adapter=fwd,
        reverse_adapter=rev,
        max_len=31
    )

    # training
    train_loader, test_loader = experiment.get_dataloader(
        min_count=min_count, use_cuda=use_cuda, batch_size=batch_size)
    device = torch.device(f"cuda:{cuda_id}" if (
        use_cuda and torch.cuda.is_available()) else "cpu")

    train_kwargs = {
        "epochs": epochs,
        "threshold": threshold,
        "device": device,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "save_dir": save_dir,
        "beta_schedule": True,
        "force_matching": True,
        "force_epochs": reg_epochs,
    }

    # evaluate model
    target_len = experiment.random_region_length
    global counter

    def objective(trial: optuna.Trial) -> float:
        global counter

        model = CNN_PHMM_VAE_FAST(
            motif_len=target_len,
            embed_size=trial.suggest_int("embed_size", 2, 10),
            hidden_size=trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256]))
        model_str = str(type(model)).split("\'")[-2].split(".")[-1].lower()
        if multi > 1:
            model_str += f"_{counter:03d}"
            counter += 1
        model_str += ".mdl"
        logger.info(f"training {model_str}")

        lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
        decay = trial.suggest_loguniform("weight_decay", 1e-3, 1e-1)

        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=decay)
        model = model.to(device)

        train_kwargs.update({
            "model": model,
            "model_str": model_str,
            "optimizer": optimizer})
        losses = models.train(**train_kwargs)
        torch.cuda.empty_cache()

        test_losses = [loss[1] for loss in losses]
        return min(test_losses)

    if storage is None:
        storage = save_dir/"optimization.db"
    study = optuna.create_study(
        study_name="chip",
        storage="sqlite:///" + str(storage.absolute()),
        direction='minimize',
        load_if_exists=True)
    study.optimize(objective, n_trials=multi)

    trial = study.best_trial

    print('test_loss: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))


if __name__ == "__main__":
    Path("./.log").mkdir(parents=True, exist_ok=True)
    formatter = '%(levelname)s : %(name)s : %(asctime)s : %(message)s'
    logging.basicConfig(
        filename='.log/logger.log',
        level=logging.DEBUG,
        format=formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    main()
