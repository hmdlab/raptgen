## run 10 motif split simulation script
import logging

import click 
import numpy as np
from pathlib import Path

import torch

from src import models
from src.models import CNN_PHMM_VAE
from src.data import SingleRound, Result

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
default_path = str(Path(f"{dir_path}/../out/gmm").resolve())

@click.command(help='select gmm center with trained model',
    context_settings=dict(show_default=True))
@click.argument("seqpath", type=click.Path(exists = True))
@click.argument("modelpath", type=click.Path(exists = True))
@click.option("--use-cuda/--no-cuda", help = "use cuda if available", is_flag=True, default = True)
@click.option("--cuda-id", help = "the device id of cuda to run", type = int, default = 0)
@click.option("--save-dir", help = "path to save results", type = click.Path(), default=default_path)
@click.option("--fwd", help = "forward adapter", type = str, default=None)
@click.option("--rev", help = "reverse adapter", type = str, default=None)
def main(seqpath, modelpath, cuda_id, use_cuda, save_dir, fwd, rev):
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
    sequences = result.get_gmm_probable_sequences()
    points = result.gmm_centers
    
    logger.info(f"saving to {save_dir}/gmm_seq.csv")
    with open(save_dir/"gmm_seq.csv","w") as f:
        f.write("gmm_index,seq,x,y\n")
        for i,(seq,(x,y)) in enumerate(zip(sequences, points)):
            logger.info(f"{seq},({x:.2f},{y:.2f})")
            f.write(f"{i},{seq},{x},{y}\n")

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