# Supplementary code for "RaptGen: A variational autoencoder with profile hidden Markov model for generative aptamer discovery"

## Tested environment

* Ubuntu == 18.04.5
* python == 3.7.9
* pytorch == 1.4.0
* cuda == 10.0

For other requirements, see [requirements.txt](requirements.txt). Also We verified that the codes are runnable in the provided Docker environment (see [Dockerfile](Dockerfile)). The requirements are installable with;

```shell
pip install -r requirements.txt
```

You also need to install `cairo` library to generate profile hmm image. For mac OS X, it can be installed by `brew install cairo && brew install pango`. For Ubuntu `sudo apt-get install -y libcairo2` would work.

## Quickstart

All scripts has `--help` command that print the options and the arguments if required. For example,

```text
% python scripts/multiple.py --help 
Usage: multiple.py [OPTIONS]

  run experiment with multiple motif

Options:
  --n-motif INTEGER       the number of motifs  [default: 10]
  --n-seq INTEGER         the number of the sequence to generate  [default:
                          10000]

  --seed INTEGER          seed for seqeunce generation reproduction  [default:
                          0]

  --error-rate FLOAT      the ratio to modify sequence  [default: 0.1]
  --epochs INTEGER        the number of training epochs  [default: 1000]
  --threshold INTEGER     the number of epochs with no loss update to stop
                          training  [default: 50]

  --use-cuda / --no-cuda  use cuda if available  [default: True]
  --cuda-id INTEGER       the device id of cuda to run  [default: 0]
  --save-dir PATH         path to save results  [default:
                          /Users/niwn/raptgen/out/simlulation/multiple]

  --reg-epochs INTEGER    the number of epochs to conduct state transition
                          regularization  [default: 50]

  --help                  Show this message and exit.  [default: False]
```

### Evaluating multi-motifs

To run the experiment with multiple sequence motifs, run;

```shell
python3 scripts/multiple.py 
```

### Evaluating paired-motifs

To run the experiment with paired sequence motifs, run;

```shell
python3 scripts/paired.py
```

### Evaluating real data

To run the experiment with sequence files, run;

```shell
python3 scripts/real.py data/sample/sample.fasta
```

`.fa`, `.fasta`, and `.fastq` files are automatically processed.

### Run GMM

To select the center of the GMM populations, run;

```shell
python3 scripts/gmm.py \
  data/sample/sample.fasta \
  data/sample/cnn_phmm_vae.mdl
```

### Run BO

To conduct multipoint Bayesian optimization, run;

```shell
python3 scripts/bo.py \
  data/real/A_4R.fastq \
  results/real/A_best.mdl \
  results/real/A_evaled.csv
```

The evaluates seuqneces should hold random region only and each row should be written in  `[string],[value]` format.

```text
AACGAGAGATGGTAGACCTATCTTTTAGCC,79.0
GTAGAGATTCTGAGGGTTCTCCTGCTATA,107.1
TTTTATAAAAAAGTGTTTAAAAAAGATTCA,-3.6
...
```

## Directory structure

```text
.
├── data
│   ├── real
│   ├── sample
│   └── simulation
│       ├── multiple
│       └── paired
├── results
│   ├── real
│   └── simulation
│       ├── multiple
│       └── paired
├── scripts
└── src
    ├── data
    ├── models
    └── visualization
```
