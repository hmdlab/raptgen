## run 10 motif split simulation script
import logging
logger = logging.getLogger(__name__)


import click 
import numpy as np
from pathlib import Path

import torch
from torch import optim

from src import models
from src.models import CNN_Mul_VAE,  LSTM_Mul_VAE,  CNNLSTM_Mul_VAE
from src.models import CNN_AR_VAE,   LSTM_AR_VAE,   CNNLSTM_AR_VAE
from src.models import CNN_PHMM_VAE, LSTM_PHMM_VAE, CNNLSTM_PHMM_VAE

from src.data import SequenceGenerator, SingleRound

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
default_path = str(Path(f"{dir_path}/../out/simlulation/multiple").resolve())

@click.command(help='run experiment with multiple motif',context_settings=dict(show_default=True))
@click.option("--n-motif", help = "the number of motifs", type = int, default = 10)
@click.option("--n-seq", help = "the number of the sequence to generate", type = int, default = 10000)
@click.option("--seed", help = "seed for seqeunce generation reproduction", type = int, default = 0)
@click.option("--error-rate", help = "the ratio to modify sequence", type = float, default = 0.1)
@click.option("--epochs", help = "the number of training epochs", type = int, default = 1000)
@click.option("--threshold", help = "the number of epochs with no loss update to stop training", type = int, default = 50)
@click.option("--use-cuda/--no-cuda", help = "use cuda if available", is_flag=True, default = True)
@click.option("--cuda-id", help = "the device id of cuda to run", type = int, default = 0)
@click.option("--save-dir", help = "path to save results", type = click.Path(), default=default_path)
@click.option("--reg-epochs", help = "the number of epochs to conduct state transition regularization", type = int, default=50)
@click.option("--multi", help = "the number of training for multiple times", type = int, default=1)
def main(n_motif, n_seq, seed, error_rate, epochs, threshold, cuda_id, use_cuda, save_dir, reg_epochs, multi):
    logger = logging.getLogger(__name__)
    logger.info(f"saving to {save_dir}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok = True, parents=True)

    # generate sequences
    fwd_adapter = "AAAAA"
    rev_adapter = "GGGGG"

    generator = SequenceGenerator(
        num_motifs = n_motif,
        seed=seed, 
        fix_random_region_length=True, 
        error_rate=error_rate, 
        generate_motifs=True, 
        add_primer=True, 
        forward_primer=fwd_adapter,
        reverse_primer=rev_adapter, 
        paired=False)
    
    reads, motif_indices = generator.sample(n_seq)
    with open(save_dir/"seqences.txt","w") as f:
        for index, read in zip(motif_indices, reads):
            f.write(f"{index}, {read}\n")
    with open(save_dir/"motifs.txt","w") as f:
        for motif in generator.motifs:
            f.write(f"{motif}\n")   

    experiment = SingleRound(
        reads,
        forward_adapter = fwd_adapter,
        reverse_adapter = rev_adapter)

    # training 
    train_loader, test_loader = experiment.get_dataloader()
    device = torch.device(f"cuda:{cuda_id}" if (use_cuda and torch.cuda.is_available()) else "cpu")
    
    train_kwargs = {
        "epochs"         : epochs,
        "threshold"      : threshold,
        "device"         : device,
        "train_loader"   : train_loader,
        "test_loader"    : test_loader,
        "save_dir"       : save_dir,
        "beta_schedule"  : True, 
        "force_matching" : True,
        "force_epochs"   : reg_epochs,
    }

    # evaluate models
    target_len = experiment.random_region_length
    cnn_mul_vae      = CNN_Mul_VAE    (target_len=target_len, embed_size=2)
    lstm_mul_vae     = LSTM_Mul_VAE   (target_len=target_len, embed_size=2)
    cnnlstm_mul_vae  = CNNLSTM_Mul_VAE(target_len=target_len, embed_size=2)
    
    cnn_phmm_vae     = CNN_PHMM_VAE    (motif_len=target_len, embed_size=2)
    lstm_phmm_vae    = LSTM_PHMM_VAE   (motif_len=target_len, embed_size=2)
    cnnlstm_phmm_vae = CNNLSTM_PHMM_VAE(motif_len=target_len, embed_size=2)

    cnn_ar_vae       = CNN_AR_VAE    (embed_size=2)
    lstm_ar_vae      = LSTM_AR_VAE   (embed_size=2)
    cnnlstm_ar_vae   = CNNLSTM_AR_VAE(embed_size=2)

    results = dict()
    for i in range(multi):
        for model in [
            cnn_phmm_vae     ,
            cnn_ar_vae       ,
            cnn_mul_vae      ,
            
            lstm_phmm_vae    ,
            lstm_ar_vae      ,
            lstm_mul_vae     ,
            
            cnnlstm_phmm_vae ,
            cnnlstm_ar_vae   ,
            cnnlstm_mul_vae  
        ]:
            model_str = str(type(model)).split("\'")[-2].split(".")[-1].lower()
            if multi > 1:
                model_str += f"_{i}"
            model_str += ".mdl"
            print (f"training {model_str}")
            
            optimizer = optim.Adam(model.parameters())
            model = model.to(device)

            train_kwargs.update({
                "model"        : model,
                "model_str"    : model_str,
                "optimizer"    : optimizer})

            results[model_str] = models.train(**train_kwargs)
            torch.cuda.empty_cache()

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