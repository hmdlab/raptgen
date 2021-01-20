## run 10 motif split simulation script
import logging

import click 
import numpy as np
import pandas as pd
from pathlib import Path

import torch

from src import models
from src.models import CNN_PHMM_VAE
from src.data import SingleRound, Result

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
default_path = str(Path(f"{dir_path}/../out/bo").resolve())

@click.command(help='run Bayesian optimization with trained model and evaluated results',
    context_settings=dict(show_default=True))
@click.argument("seqpath", type=click.Path(exists=True))
@click.argument("modelpath", type=click.Path(exists = True))
@click.argument("evalpath", type=click.Path(exists = True))
@click.option("--use-cuda/--no-cuda", help = "use cuda if available", is_flag=True, default = True)
@click.option("--cuda-id", help = "the device id of cuda to run", type = int, default = 0)
@click.option("--fwd", help = "forward adapter", type = str, default=None)
@click.option("--rev", help = "reverse adapter", type = str, default=None)
@click.option("--save-dir", help = "path to save results", type = click.Path(), default=default_path)
def main(seqpath, modelpath, evalpath, cuda_id, use_cuda, save_dir, fwd, rev):
    logger = logging.getLogger(__name__)
    
    logger.info(f"saving to {save_dir}")
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok = True, parents=True)

    experiment = SingleRound(
        path = seqpath,
        forward_adapter = fwd,
        reverse_adapter = rev)
    target_len = experiment.random_region_length
    model = CNN_PHMM_VAE(target_len, embed_size=2)
    device = torch.device(f"cuda:{cuda_id}" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model.load_state_dict(torch.load(modelpath, map_location=device))

    result = Result(
        model,
        experiment=experiment,
        path_to_save_results=save_dir,
        load_if_exists=True
    )    

    df = pd.read_csv(evalpath,names=["seq", "act"])
    seq, act = df.values.T    

    result.evaluated_X = result.embed_sequences(seq)
    result.evaluated_y = -act[:,None]
    locations = result.get_bo_result(force_rerun=True)
    scores = result._points_to_score(torch.from_numpy(locations).float())
    probable_sequences = list(zip(*scores))[1]
    reembed_positions = result.embed_sequences(probable_sequences)
    
    logger.info(f"saving to {save_dir}/bo_seq.csv")
    with open(save_dir/"bo_seq.csv","w") as f:
        f.write("bo_index,seq,x,y,re_x,re_y\n")
        for i,(seq,(x,y),(re_x,re_y)) in enumerate(zip(probable_sequences,locations, reembed_positions)):
            logger.info(f"{seq},({x:.2f},{y:.2f})->({re_x:.2f},{re_y:.2f})")
            f.write(f"{i},{seq},{x},{y},{re_x},{re_y}\n")

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