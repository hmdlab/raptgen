#!/usr/bin/env python
# coding: utf-8
from collections import Counter, defaultdict
import re
import datetime
import torch
import numpy as np
import pickle
from pathlib import Path
from enum import IntEnum
from matplotlib import pyplot as plt
from itertools import product
import pandas as pd
import matplotlib
from tqdm.auto import tqdm
from collections import Counter
from multiprocessing import Pool

import logging
logger = logging.getLogger(__name__)


class State(IntEnum):
    M = 0
    I = 1
    D = 2

class Transition(IntEnum):
    M2M = 0
    M2I = 1
    M2D = 2
    I2M = 3
    I2I = 4
    D2M = 5
    D2D = 6

class nt_index(IntEnum):
    A = 0
    T = 1
    G = 2
    C = 3
    PAD = 4
    SOS = 5
    EOS = 6
    U = 1

def one_hot_index(seq):
    return [int(nt_index[char]) for char in seq]


def one_hot_encode(nucleotide, padding=0):
    """入力された文字列に対してOne-hotなnp形式を返す
    """
    # パディングの大きさを指定する
    arr = np.vstack((np.eye(4), np.ones(4)[None, :]*0.25))
    return arr[one_hot_index("N"*padding + nucleotide + "N"*padding)].T





class SNV(IntEnum):
    Mutation = 0
    Insertion = 1
    Deletion = 2


class SequenceGenerator():
    def __init__(self, num_motifs=1, motif_length=10, motifs=None,
                 target_length=20, fix_random_region_length=True, error_rate=0, generate_motifs=True, middle_insert_range=[2, 6],
                 seed=0, add_primer=True, forward_primer="AAAAA", reverse_primer="GGGGG", one_side_proba=0.5, paired=False):
        np.random.seed(seed)

        if generate_motifs:
            self.motifs = ["".join(np.random.choice(
                list("ATGC"), motif_length)) for _ in range(num_motifs)]
        else:
            self.motifs = motifs

        self.error_indices = 1 + \
            np.argsort(np.random.random(size=motif_length-1))[:3]
        self.mut_idx, self.ins_idx, self.del_idx = self.error_indices

        logger.info(f"error rate is {error_rate*100:.1f}%")
        for idx, motif in enumerate(self.motifs):
            seq = [ch for ch in motif]
            mut = self.mutate(seq[self.mut_idx])
            if error_rate != 0:
                seq[self.mut_idx] = f"[{seq[self.mut_idx]}>{mut}]"
                seq[self.ins_idx] = f"[+]{seq[self.ins_idx]}"
                seq[self.del_idx] = f"{seq[self.del_idx].lower()}"
            seq = "".join(seq)
            logger.info(f"motif {idx} is {seq}")

        self.num_motifs = num_motifs
        self.error_rate = error_rate
        self.target_length = target_length
        self.forward_primer = forward_primer
        self.reverse_primer = reverse_primer
        self.add_primer = add_primer

        self.one_side_proba = one_side_proba
        self.middle_insert_range = middle_insert_range
        self.paired = paired

    def mutate(self, char):
        return "TGCA"["ATGC".index(char)]

    def sample_motif(self, n):
        motif_indices = np.random.randint(self.num_motifs, size=n)
        has_errors = np.random.random(size=n) < self.error_rate
        # mutation, insertion, deletion
        error_types = np.random.choice(SNV, size=n)
        sequences = []
        for motif_index, has_error, error_type in zip(motif_indices, has_errors, error_types):
            motif = self.motifs[motif_index]
            seq = [ch for ch in motif]
            if has_error:
                if error_type == SNV.Mutation:
                    seq[self.mut_idx] = self.mutate(seq[self.mut_idx])
                elif error_type == SNV.Insertion:
                    seq[self.ins_idx] = np.random.choice(
                        list("ATGC")) + seq[self.ins_idx]
                elif error_type == SNV.Deletion:
                    seq[self.del_idx] = ""
                else:
                    raise NotImplementedError
            seq = "".join(seq)
            sequences.append(seq)
        return sequences, motif_indices.tolist()

    def sample(self, n=1, with_indices=True):
        motifs, motif_indices = self.sample_motif(n)
        sequences = []
        paired_indices = []
        for seq in motifs:
            if self.paired:
                seq, idx = self.insert_in_the_middle(
                    seq, nrange=self.middle_insert_range, one_side_proba=self.one_side_proba)
                paired_indices += [idx]
            random_region = "".join(np.random.choice(
                list("ATGC"), size=self.target_length-len(seq)))
            l = np.random.randint(len(random_region))
            if self.add_primer:
                sequences.append(
                    self.forward_primer + random_region[:l] + seq + random_region[l:] + self.reverse_primer)
            else:
                sequences.append(random_region[:l] + seq + random_region[l:])

        if self.paired and with_indices:
            return sequences, motif_indices, paired_indices
        elif with_indices:
            return sequences, motif_indices
        return sequences

    def insert_in_the_middle(self, sequence, nrange=[2, 6], one_side_proba=0.5):
        n = np.random.randint(*nrange)
        if np.random.random() < one_side_proba:
            if np.random.choice(["l", "r"]) == "l":
                l_motif = sequence[:len(sequence)//2]
                r_motif = ""
                idx = 1
            else:
                l_motif = ""
                r_motif = sequence[len(sequence)//2:]
                idx = 2
        else:
            l_motif = sequence[:len(sequence)//2]
            r_motif = sequence[len(sequence)//2:]
            idx = 0
        return l_motif + "".join(np.random.choice(list("ATGC"), size=n)) + r_motif, idx


def get_reads_with_id_prefix(path, prefix_on, prefix_off):
    reads = []
    read = ""
    switch = False
    with path.open() as f:
        for line in f.readlines():
            if line[0] == prefix_off:
                switch = False
                reads.append(read)
                read = ""
            if switch:
                read = read + line.strip()
            if line[0] == prefix_on:
                switch = True
                read = ""
    return reads


def read_fasta(path):
    return get_reads_with_id_prefix(Path(path), ">", ">")


def read_fastq(path):
    return get_reads_with_id_prefix(Path(path), "@", "+")


class SingleRound:
    """pass path or raw_reads to make class of selex experiment per round.
    """

    def __init__(self, raw_reads: list = None, forward_adapter=None, reverse_adapter=None, name=None, tolerance=0, path: str = None):
        assert path is not None or raw_reads is not None, "either path or raw_reads has to be specified"
        if path:
            path = Path(path)
            if path .suffix == ".fastq":
                logger.info("reading fastq format sequence")
                raw_reads = read_fastq(path)
            elif path.suffix in {".fasta",".fa"} :
                logger.info("reading fasta format sequence")
                raw_reads = read_fasta(path)
            else:
                logger.critical("please specify a file with fasta or fastq format")
                quit()

        self.raw_reads = raw_reads
        self.calc_target_length()

        if forward_adapter is None or reverse_adapter is None:
            logger.info("adapter info not provided. estimating value")
            self.calc_experimental_settings()
        else:
            logger.info(f"sequence design : {forward_adapter}-[random]-{reverse_adapter}")
            self.set_adapters(forward_adapter, reverse_adapter)
        
        if name:
            self.name = name
        else:
            self.name = re.sub(r'[-\.\:]', "",
                               str(datetime.datetime.now())).replace(" ", "_")
        logger.info(f"experiment name : {self.name}")
        self.tolerance = tolerance

    def get_adapters(self):
        return self.forward_adapter, self.reverse_adapter

    def set_adapters(self, forward_adapter: str, reverse_adapter: str):
        self.forward_adapter = forward_adapter
        self.forward_adapter_length = len(forward_adapter)

        self.reverse_adapter = reverse_adapter
        self.reverse_adapter_length = len(reverse_adapter)

        self.random_region_length = self.target_length - \
            self.reverse_adapter_length - self.forward_adapter_length

    def calc_target_length(self):
        from collections import Counter, defaultdict
        self.read_counter = Counter(self.raw_reads)

        # calc most common length
        d = defaultdict(int)
        for key, value in self.read_counter.items():
            d[len(key)] += value
        self.target_length = sorted(d.items(), key=lambda x: -x[1])[0][0]

    def calc_experimental_settings(self):
        """calculate sequence adapters in a heuristic way
        """

        # fwd
        max_count = None
        est_adapter = ""
        for i in range(1, self.target_length):
            d = defaultdict(int)
            for seq, count in self.read_counter.most_common():
                if len(seq) < i or len(d) > 100 and seq[:i] not in d.keys():
                    continue
                d[seq[:i]] += count
            top_seq, top_count = sorted(d.items(), key=lambda x: -x[1])[0]
            if max_count is not None and top_count < max_count * 0.5:  # heuristics
                logger.info(
                    f"estimated forward adapter len is {i-1} : {est_adapter}")
                break
            max_count = sorted(d.items(), key=lambda x: -x[1])[0][1]
            est_adapter = top_seq
        fwd_len = i - 1
        fwd_adapter = est_adapter

        # rev
        max_count = None
        est_adapter = ""
        for i in range(1, self.target_length):
            d = defaultdict(int)
            for seq, count in self.read_counter.most_common():
                if len(seq) < i or len(d) > 100 and seq[-i:] not in d.keys():
                    continue
                d[seq[-i:]] += count
            top_seq, top_count = sorted(d.items(), key=lambda x: -x[1])[0]
            if max_count is not None and top_count < max_count * 0.5:  # heuristics
                logger.info(
                    f"estimated reverse adapter len is {i-1} : {est_adapter}")
                break
            max_count = sorted(d.items(), key=lambda x: -x[1])[0][1]
            est_adapter = top_seq
        rev_len = i - 1
        rev_adapter = est_adapter

        rand_len = self.target_length - rev_len - fwd_len

        logger.info(
            f"filtering with : {fwd_adapter}({fwd_len}N)-{rand_len}N-{rev_adapter}({rev_len}N)")

        # write estimated experimental settings
        self.set_adapters(fwd_adapter, rev_adapter)

    def get_sequences_and_count(self):
        c = Counter(self.raw_reads)
        return c.most_common()

    def get_filter_passed_sequences_and_count(self, random_only=False):
        if random_only:
            return {self.cut_adapters(key): value for key, value in self.get_sequences_and_count()}
        else:
            c = Counter(self.get_filter_passed_sequences())
            return c.most_common()

    def filter_function(self, read):
        has_forward = read[: self.forward_adapter_length] == self.forward_adapter
        has_reverse = read[-self.reverse_adapter_length:] == self.reverse_adapter
        match_random_region_len = abs(len(read) - self.target_length) <= self.tolerance
        return has_forward and has_reverse and match_random_region_len

    def get_filter_passed_sequences(self, random_only=False):
        self.filter_passed = list(filter(self.filter_function, self.raw_reads))
        if random_only:
            return [self.cut_adapters(read) for read in self.filter_passed]
        return self.filter_passed

    def cut_adapters(self, seq):
        return seq[self.forward_adapter_length:-self.reverse_adapter_length]

    def __str__(self):
        return f"experiment of {len(self.raw_reads)} raw reads"

    def get_dataloader(self, min_count=1, test_size=0.1, batch_size=512, shuffle=True, use_cuda = True):
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader
        
        self.min_count = min_count
        kwargs = {'num_workers': 1, 'pin_memory': True} if (use_cuda and torch.cuda.is_available()) else {}
        # load RAPT1-4R and filter reads to count>1, then make it to one hot encoded tensor
        c = self.get_filter_passed_sequences(random_only=True)
        sequences = list(
            filter(lambda seq_count: seq_count[1] >= min_count, Counter(c).most_common()))
        seq, _ = zip(*sequences)

        train_test = np.array(list(map(one_hot_index, seq)))
        logger.info(f"# of sequences -> {len(train_test)}")
        train_data, test_data = train_test_split(
            train_test, test_size=test_size, shuffle=shuffle)
        train_data = torch.from_numpy(train_data).long()
        test_data = torch.from_numpy(test_data).long()
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,  **kwargs)
        test_loader = DataLoader(
            test_data,  batch_size=batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        out_data = self.data[index]
        return out_data

def local_alignment(s1, s2, print_result=False, global_alignment=False):
    GAP_COST = -1
    MATCH_COST = +1
    MISMATCH_COST = -1

    # [[(0,0) for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    dp = np.zeros((len(s1) + 1, len(s2) + 1, 2), dtype=np.int)

    def s(c1, c2):
        return MATCH_COST if c1 == c2 else MISMATCH_COST

    def idx_max(*args):
        argmax_i = np.argmax(args)
        return argmax_i, args[argmax_i]

    def concat(ret, a, b):
        return a + ret[0], b + ret[1]

    OBJ = 1
    POINTER_INDEX_TO_COND = ["MATCH", "S1_GAP", "S2_GAP", "END"]

    # DP
    for i1, c1 in enumerate(s1):
        for i2, c2 in enumerate(s2):
            l = [ dp[i1][i2][OBJ] + s(c1, c2),
                dp[i1][i2 + 1][OBJ] + GAP_COST,
                dp[i1 + 1][i2][OBJ] + GAP_COST]
            if not global_alignment :
                l += [0]
            dp[i1 + 1][i2 + 1] = idx_max( *l )

    # local_alignment_traceback
    # logger.info(list(zip(*np.where(dp[:,:,OBJ]==dp[:,:,OBJ].max()))))
    result = []
    if global_alignment:
        traceback_starts = [(len(s1), len(s2))]
    else:
        traceback_starts = list(zip(*np.where(dp[:, :, OBJ] == dp[:, :, OBJ].max())))
    for i1, i2 in traceback_starts:
        traceback_pointer = dp[i1, i2, 0]
        ret = ["", ""]
        while True:
            if print_result:
                logger.info(i1, i2, POINTER_INDEX_TO_COND[traceback_pointer])
            if POINTER_INDEX_TO_COND[traceback_pointer] == "MATCH":
                i1 -= 1
                i2 -= 1
                if i1 < 0 or i2 < 0:
                    break
                if s1[i1] != s2[i2]:
                    ret = concat(ret, s1[i1].lower(), s2[i2].lower())
                else:
                    ret = concat(ret, s1[i1], s2[i2])
                traceback_pointer = dp[i1, i2, 0]
            elif POINTER_INDEX_TO_COND[traceback_pointer] == "S1_GAP":
                i1 -= 1
                if i1 < 0 or i2 < 0:
                    break
                ret = concat(ret, s1[i1], "-")
                traceback_pointer = dp[i1, i2, 0]
            elif POINTER_INDEX_TO_COND[traceback_pointer] == "S2_GAP":
                i2 -= 1
                if i1 < 0 or i2 < 0:
                    break
                ret = concat(ret, "-", s2[i2])
                traceback_pointer = dp[i1, i2, 0]
            else:
                break
        if print_result:
            logger.info(ret, dp[:, :, OBJ].max())
        result.append(ret)
    return dp[:, :, OBJ].max(), result


# from https://rosettacode.org/wiki/Levenshtein_distance#Memoized_recursive_version_2
from functools import lru_cache
@lru_cache(maxsize=2**26)
def edit_distance(s, t):
    if not s:
        return len(t)
    if not t:
        return len(s)
    if s[0] == t[0]:
        return edit_distance(s[1:], t[1:])
    l1 = edit_distance(s, t[1:])
    l2 = edit_distance(s[1:], t)
    l3 = edit_distance(s[1:], t[1:])
    return 1 + min(l1, l2, l3)


def get_complement_sequence(seq):
    return "".join(reversed([list("ATGC")["TACG".index(char)] for char in seq]))


class ProfileHMMSampler():
    def __init__(self, transition_proba, emission_proba, proba_is_log=False):
        self.e = emission_proba
        self.a = transition_proba
        if proba_is_log:
            self.e = np.exp(self.e)
            self.a = np.exp(self.a)
        self.e = self.e / np.sum(self.e, axis=1)[:, None]

    def sample(self, sequence_only=False, debug=False):
        idx, state = (0, State.M)
        states = [(idx, state)]
        seq = ""
        while True:
            if state == State.M:
                p = self.a[idx][np.array([
                    Transition.M2M.value,
                    Transition.M2I.value,
                    Transition.M2D.value])]
            elif state == State.I:
                p = np.stack([
                    self.a[idx][Transition.I2M.value],
                    self.a[idx][Transition.I2I.value],
                    0])
            elif state == State.D:
                p = np.stack([
                    self.a[idx][Transition.D2M.value],
                    0,
                    self.a[idx][Transition.D2D.value]])
            else:
                logger.info("something wrong")

            state = np.random.choice([State.M, State.I, State.D], p=p/sum(p))
            if state != State.I:
                idx += 1
            states.append((idx, state))
            if idx == self.a.shape[0]:
                break

            if state == State.M:
                # logger.info("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(*self.e[idx-1]))

                seq += np.random.choice(list("ATGC"), p=self.e[idx-1])
                if debug:
                    logger.info(idx, state, self.e[idx-1], seq[-1])
            elif state == State.I:
                seq += np.random.choice(list("atgc"))
            else:
                seq += "_"
        if not sequence_only:
            return states, seq
        else:
            return seq

    def most_probable(self, sequence_only=False):
        idx, state = (0, State.M)
        states = [(idx, state)]
        seq = ""
        while True:
            if state == State.M:
                p = self.a[idx][np.array([
                    Transition.M2M.value,
                    Transition.M2I.value,
                    Transition.M2D.value])]
            elif state == State.I:
                p = [
                    self.a[idx][Transition.I2M.value],
                    0,
                    0]
            elif state == State.D:
                p = [
                    self.a[idx][Transition.D2M.value],
                    0,
                    self.a[idx][Transition.D2D.value]]
            else:
                logger.info("something wrong")
            p[np.argmax(p)] += 1000000
            state = np.random.choice([State.M, State.I, State.D], p=p/sum(p))
            if state != State.I:
                idx += 1
            states.append((idx, state))

            if idx == self.a.shape[0]:
                break

            if state == State.M:
                # logger.info("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(*self.e[idx-1]))
                p = np.copy(self.e[idx-1])
                p[np.argmax(p)] += 100000
                seq += np.random.choice(list("ATGC"), p=p/sum(p))
            elif state == State.I:
                seq += "N"
            else:
                seq += "_"
        if not sequence_only:
            return states, seq
        else:
            return seq

    def calc_seq_proba(self, seq: str):
        one_hot_seq = torch.tensor(one_hot_index(seq))
        model_len = self.e.shape[0]
        random_len = len(seq)

        e = np.log(self.e)
        a = np.log(self.a)

        F = torch.ones((3, model_len + 2, random_len + 1)) * (-100)

        # init
        F[0, 0, 0] = 0

        for i in range(random_len + 1):
            for j in range(model_len + 1):
                # State M
                if j*i != 0:
                    F[State.M, j, i] = e[j - 1][one_hot_seq[i - 1]] + \
                        torch.logsumexp(torch.stack((
                            a[j - 1, Transition.M2M] +
                            F[State.M, j - 1, i - 1],
                            a[j - 1, Transition.I2M] +
                            F[State.I, j - 1, i - 1],
                            a[j - 1, Transition.D2M] + F[State.D, j - 1, i - 1])), dim=0)

                # State I
                if i != 0:
                    F[State.I, j, i] = - 1.3863 + \
                        torch.logsumexp(torch.stack((
                            a[j, Transition.M2I] + F[State.M, j, i-1],
                            a[j, Transition.I2I] + F[State.I, j, i-1]
                        )), dim=0)

                # State D
                if j != 0:
                    F[State.D, j, i] = \
                        torch.logsumexp(torch.stack((
                            a[j - 1, Transition.M2D] + F[State.M, j - 1, i],
                            a[j - 1, Transition.D2D] + F[State.D, j - 1, i]
                        )), dim=0)

        F[State.M, model_len+1, random_len] = \
            torch.logsumexp(torch.stack((
                a[model_len, Transition.M2M] +
                F[State.M, model_len, random_len],
                a[model_len, Transition.I2M] +
                F[State.I, model_len, random_len],
                a[model_len, Transition.D2M] +
                F[State.D, model_len, random_len]
            )), dim=0)

        return F[State.M, model_len+1, random_len]


class Result():
    """実験結果の保存のためのクラス"""
    from src.visualization import provide_ax

    def __init__(self,
                 model,
                 path_to_selex: str = None,
                 experiment: SingleRound = None,
                 path_to_result_csv: str = None,
                 path_to_model: str = None,
                 lazy_mu_eval=False,
                 path_to_save_results=None,
                 evaluated_X=None,
                 evaluated_y=None,
                 load_if_exists=False,
                 min_count = 1
                 ):

        if experiment is None:
            self.experiment = SingleRound(path=Path(path_to_selex))
        else:
            self.experiment = experiment

        if path_to_result_csv:
            self.result_df = pd.read_csv(Path(path_to_result_csv))
        else:
            logger.info("skip loading training result")

        if path_to_model:
            model.load_state_dict(torch.load(path_to_model))
        else:
            logger.info("skip loading model parameters")

        self.model = model

        if path_to_save_results is None:
            self.path_to_save_results = Path(
                "result_" + self.get_result_hash())
        else:
            self.path_to_save_results = Path(path_to_save_results)
        if load_if_exists:
            self.path_to_save_results.mkdir(parents=True, exist_ok=True)
        else:
            self.path_to_save_results.mkdir(parents=True)

        self.is_phmm = "PHMM" in str(model.__class__)
        self.min_count = min_count
        if not lazy_mu_eval:
            logger.info("evaluating mu")
            self.get_mean_vectors_from_experiment()

        self.evaluated_X = evaluated_X
        self.evaluated_y = evaluated_y

    def get_mean_vectors_from_experiment(self, get_raw_seq=False, force=False):
        if not hasattr(self, "mus") or not hasattr(self, "seqs") or force:
            loaders = self.experiment.get_dataloader(
                shuffle=False, min_count=self.min_count)
            with torch.no_grad():
                self.model.eval()
                mus = []
                seqs = []
                for loader in loaders:
                    for data in loader:
                        _, mu, logvar = self.model(data, deterministic=True)
                        mus += [*mu.detach().numpy()]
                        for datum in data:
                            seq = "".join(np.array(list("ATGC"))
                                          [datum.numpy()])
                            seqs.append(seq)

            self.mus = np.stack(mus)
            self.seqs = seqs

        if get_raw_seq:
            return self.mus, self.seqs
        return self.mus

    def get_result_hash(self):
        # modelのパラメタが一意に決まっていてhashしやすいのでこれを利用する
        import hashlib
        b = "".join(["{:.2f}".format(i.flatten()[0].cpu().detach().numpy())
                     for i in self.model.parameters()])
        self.hash = hashlib.sha1(b.encode()).hexdigest()[:10]
        logger.info(f"hash : {self.hash}")
        return self.hash

    def calc_gmm(self, dim=10, calc_times=100, force=False):
        from sklearn.mixture import GaussianMixture
        logger.info("calculating gmm centers")
        X = self.mus
        gmm_path = self.path_to_save_results/"gmm.pkl"
        if gmm_path.exists():
            logger.info(f"loading {gmm_path}")
            with gmm_path.open("rb") as f:
                best_gmm = pickle.load(f)
                best_aic = best_gmm.aic(X)
        else:
            best_aic = np.inf
            pbar = tqdm(range(calc_times))
            for i in pbar:
                gmm = GaussianMixture(dim, covariance_type="full").fit(X)
                if gmm.aic(X) < best_aic:
                    best_aic = gmm.aic(X)
                    best_gmm = gmm
                pbar.set_description(
                    "[" + "⠸⠴⠦⠇⠋⠙"[i % 6] + "]" + f"{best_aic:.2f}")

            with gmm_path.open("wb") as f:
                pickle.dump(best_gmm, f)

        logger.info(f"best aic : {best_aic}")
        self.gmm = best_gmm
        self.aic = best_aic
        self.gmm_classes = best_gmm.predict(X)
        self.gmm_centers = best_gmm.means_

    def embed_sequences(self, sequences):
        """
        docstring
        """
        if type(sequences) == str:
            sequences = [sequences]

        # https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180
        model_device = next(self.model.parameters()).device
        with torch.no_grad():
            self.model.eval()
            mus = []
            for sequence in sequences:
                recon, mu, logvar = self.model(
                    torch.Tensor(
                        [one_hot_index(sequence)],
                        device=model_device).long())
                mus += [mu]
        return torch.cat(mus)

    @provide_ax
    def plot_gmm(self, ax, fig=None, save=True,no_colors=False,no_gmm_centers = False):
        if not hasattr(self, "gmm"):
            logger.info("calculating gmm")
            self.calc_gmm()
        if len(self.gmm_classes) > 1000:
            if no_colors:
                ax.scatter(*self.mus[:1000].T, s=2, c="silver")
            else:
                ax.scatter(*self.mus[:1000].T, c=self.gmm_classes[:1000], s=2)
        else:
            if no_colors:
                ax.scatter(*self.mus.T, s=2, c="silver")
            else:
                ax.scatter(*self.mus.T, c=self.gmm_classes, s=2)
        if not no_gmm_centers:
            ax.scatter(*self.gmm_centers.T, c="r", marker="*",
                    s=10, zorder=50, label="gmm center")
            for i, (x, y) in enumerate(self.gmm_centers):
                ax.text(x, y, f" {i}", ha="left", va="center", c="r", bbox=dict(
                    facecolor='white', alpha=0.5), zorder=40)

        ax.axis("square")
        if save:
            fig.savefig(self.path_to_save_results/'gmm.png')
        return ax

    @provide_ax
    def plot_means(self, ax, fig=None, with_count=False, save=True, meshgrid=True):
        if self.mus is None:
            logger.info("running estimation")
            self.get_mean_vectors_from_experiment()

        if with_count:
            c = Counter(
                self.experiment.get_filter_passed_sequences(random_only=True))
            counts = [np.log2(c[seq]) for seq in self.seqs]
            XY = self.mus[np.argsort(counts)]
            if meshgrid:
                XY_ = []
                for xy, cnt in zip(XY, [c[seq] for seq in self.seqs]):
                    XY_ += [xy]*int(np.log2(cnt))
                heatmap, xedges, yedges = np.histogram2d(
                    * np.stack(XY_).T, bins=100, range=((-3, 3), (-3, 3)))
            else:
                cs = ax.scatter(*XY.T, s=2, c=sorted(counts), vmin=0)
                cbar = fig.colorbar(cs)
                cbar.ax.set_title("log2 counts")
                legend = ax.legend(*cs.legend_elements(num=6),
                                   loc="lower right", title="log2 count")
                ax.add_artist(legend)

        else:
            ax.scatter(*self.mus.T, s=2)
        ax.axis("square")

        if save:
            fig.savefig(self.path_to_save_results/'means.png')
        return ax

    @provide_ax
    def plot_bo(self, ax, fig=None, n_grid=101, save=True, with_index=True, plot_range=(-2,2)):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        if not hasattr(self, "next_locations"):
            self.get_bo_result()

        self.bo_grid_x, self.bo_grid_y = np.meshgrid(
            np.linspace(*plot_range, n_grid),
            np.linspace(*plot_range, n_grid))
        self.bo_X = np.stack(map(np.ravel, (self.bo_grid_x, self.bo_grid_y))).T

        mu, sigma = self.bo.model.predict(self.bo_X)
        self.bo_mu = mu
        self.bo_sigma = sigma

        cont = ax.contour(self.bo_grid_x, self.bo_grid_y, -
                          (self.bo_mu-self.bo_sigma).reshape(n_grid, n_grid))
        cont.clabel(fmt='%1.1f', fontsize=8)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        ax.scatter(*self.evaluated_X.T,
                   c=self.evaluated_y[:, 0], cmap="bwr_r", ec="grey", lw=0.5, zorder=40, label="evaluated")
        if with_index:
            for i, (x, y) in enumerate(self.evaluated_X):
                ax.text(x, y, " "+str(i), color="blue", va="center")

        ax.scatter(*self.next_locations.T, marker="*",
                   color="k", zorder=50, label="bo proposed")
        if with_index:
            for i, (x, y) in enumerate(self.next_locations):
                ax.text(x, y, f" {i}", color="k",zorder=51,va = "center")
        fig.colorbar(cont, cax=cax)

        return ax

    def get_bo_result(self, n=10, domain=(-2, 2), force_rerun=False):
        if hasattr(self, "next_locations") and not force_rerun:
            return self.next_locations
        import GPyOpt
        bo_path = self.path_to_save_results/"bo.pkl"
        if bo_path.exists() and not force_rerun:
            logger.info(f"loading {bo_path}")
            with bo_path.open("rb") as f:
                self.bo = pickle.load(f)
        else:
            logger.info("calculating bo")
            assert self.evaluated_X is not None and self.evaluated_y is not None,\
                "(N, d) array: `evaluated_X` and (N, 1) array: `evaluated_y` should be set"
            self.domain = domain
            self.constraints = [{"name": f"var_{i+1}",
                                "type": "continuous",
                                "domain": self.domain}
                                for i in range(self.evaluated_X.shape[1])]

            self.bo = GPyOpt.methods.BayesianOptimization(
                None,
                domain=self.constraints,
                model_type='GP',
                acquisition_type='LCB',
                evaluator_type="local_penalization",
                batch_size=n,
                X=self.evaluated_X,
                Y=self.evaluated_y)
            with bo_path.open("wb") as f:
                pickle.dump(self.bo, f)

        self.next_locations = self.bo.suggest_next_locations()
        return self.next_locations

    

    def _points_to_score(self, points, eval_max=256):
        a, e_m = self.model.decoder(points)
        a = a.detach().numpy()
        e_m = e_m.detach().numpy()

        logger.info (f"calculating most probable sequences up to {eval_max} candidates")
        pbar = tqdm(range(len(points)))
        scores = []
        for j in pbar:
            sampler = ProfileHMMSampler(a[j], e_m[j], proba_is_log=True)
            seq_pattern = sampler.most_probable()[1].replace(
                "_", "").replace("N", "*")
            products = product(*[list("ATGC")
                                 for _ in range(seq_pattern.count("*"))])

            rets = []
            for nt_set in products:
                ret = ""
                for part, nt in zip(seq_pattern.split("*"), list(nt_set)+[""]):
                    ret += part+nt
                rets += [ret]
            if len(rets) > eval_max:
                rets = [rets[idx] for idx in np.argsort(
                    np.random.randn(len(rets)))[:eval_max]]
            with Pool() as p:
                probas = p.map(sampler.calc_seq_proba, rets)

            most_probable_seq, min_value = sorted(
                list(zip(rets, probas)), key=lambda x: x[1])[0]
            min_value = min_value.item()
            scores += [(seq_pattern, most_probable_seq, min_value)]
        self.scores = scores
        return scores

    def _save_scores(self, scores, model_type, filename, id_header="", force=False, loc=None):
        from datetime import datetime
        if filename is not None:
            assert not ((not force) and (self.path_to_save_results /
                                         filename).exists()), "file exists. to override, try: 'force=True'"
            with open(self.path_to_save_results / filename, "w") as f:
                f.write("id,method,max_model,max_seq,log_proba")
                if loc is not None:
                    f.write(",pos_x,pos_y\n")
                else:
                    f.write("\n")

                for i, (seq_pattern, most_probable_seq, min_value) in enumerate(scores):
                    id_str = datetime.now().strftime(
                        "%-y%m%d_") + id_header + f"_{i}"
                    f.write(
                        f"{id_str},{model_type},{seq_pattern},{most_probable_seq},{min_value}")
                    if loc is not None:
                        f.write(f",{loc[i][0]},{loc[i][1]}\n")
                    else:
                        f.write("\n")
            logger.info(f"saved to {self.path_to_save_results}/{filename}")

    def get_gmm_probable_sequences(self, filename=None):
        if hasattr(self, "scores"):
            return [most_probable for seq_pattern, most_probable, min_value in self.scores]
        if not hasattr(self, "gmm_centers"):
            self.calc_gmm()

        scores = self._points_to_score(torch.Tensor(self.gmm_centers))

        self._save_scores(scores, "GMM", filename)

        return [most_probable for seq_pattern, most_probable, min_value in scores]

    def plot_training_result(self, nwarmup=100, save=True, fig=None, axes = None):
        from src.visualization import get_ax
        if axes is not None and fig is not None:
            ax,ay = axes
        fig, (ax, ay) = get_ax(row_col=(2, 1), return_fig=True)

        # for mean plot
        length = 20
        v = np.sin(np.arange(length)/(length-1)*np.pi)
        v /= sum(v)

        i = 0
        for arr_name in ["test_loss", "test_recon", "train_loss"]:
            cmap = plt.get_cmap("Paired")
            test_losses = np.array(self.result_df[arr_name])
            epochs = np.arange(len(test_losses))
            conv_indices, conv_values = np.array(
                epochs[:-length+1])+length//2, np.convolve(test_losses, v, mode='valid')
            ax.plot(test_losses, c=cmap(i))

            ax.plot(conv_indices, conv_values, c=cmap(i+1), label=arr_name)
            ax.plot(conv_indices[-1], conv_values[-1], marker='.', c=cmap(i+1))
            ax.text(np.argmin(test_losses) + 1, np.min(test_losses),
                    f"←{np.min(test_losses):.2f}", ha="left", va="top", c=cmap(i+1), rotation=-45, fontsize=8,
                    bbox=dict(ec=(1, 1, 1, 1), facecolor="w", alpha=1, pad=0))
            i += 2

        for arr_name in ["test_kld"]:
            cmap = plt.get_cmap("Paired")
            test_losses = np.array(self.result_df[arr_name])
            epochs = np.arange(len(test_losses))
            conv_indices, conv_values = np.array(
                epochs[:-length+1])+length//2, np.convolve(test_losses, v, mode='valid')
            ay.plot(test_losses, c=cmap(i))

            ay.plot(conv_indices, conv_values, c=cmap(i+1), label=arr_name)
            ay.plot(conv_indices[-1], conv_values[-1], marker='.', c=cmap(i+1))

            i += 2

        min_test_loss = min(self.result_df.test_loss)
        min_test_recon = min(self.result_df.test_recon)
        dloss = abs(min_test_loss-min_test_recon)
        ax.set_ylim(min_test_recon-dloss*0.6, min_test_loss+dloss*0.6)
        ay.set_ylim(-dloss*0.1, dloss*2.1)

        ax.plot((nwarmup, nwarmup), (0, 100), "--",
                c="gray", label="profile hmm warmup")
        ay.plot((nwarmup, nwarmup), (0, 100), "--",
                c="gray", label="profile hmm warmup")
        ay.set_xlabel("epochs")
        ax.set_ylabel("loss")
        ay.set_ylabel("loss")

        ax.legend()
        ay.legend()
        ax.set_title("training result")
        if save:
            fig.savefig(self.path_to_save_results/'training.png')


class Experiments():
    """whole selex experiment"""

    def __init__(self, read_paths: list, has_same_adapters=True, k=3):
        adapters = None
        self.rounds = []
        for read_path in sorted(read_paths):
            logger.info(f"reading ... {read_path}")
            path = Path(read_path)
            if adapters is not None and has_same_adapters:
                fwd, rev = adapters
                single_round = SingleRound(path=path, name=path.stem,
                                           forward_adapter=fwd, reverse_adapter=rev)
            else:
                single_round = SingleRound(path=path, name=path.stem)
                adapters = single_round.get_adapters()
            self.rounds += [single_round]
        self.k = k
        self.kmer_list = sorted(list("".join(l)
                                     for l in product(list("ATGC"), repeat=k)))

    def has_exact_match(self, sequences):
        if type(sequences) is str:
            sequences = [sequences]
        results = dict()
        for sequence in sequences:
            appeared = set()
            for experiment in self.rounds:
                if sequence in set(experiment.get_filter_passed_sequences(random_only=True)):
                    appeared |= {experiment.name}
            results[sequence] = appeared
        return results

    def kmer(self, seq):
        return [seq[i: i+self.k] for i in range(len(seq)-self.k+1)]

    def kmer_count(self, seq, to_list=False):
        from collections import Counter
        c = Counter(self.kmer(seq))
        if to_list:
            return [c[kmer] for kmer in self.kmer_list]
        return c

    def save_frequencies(self, save_path, sequences, min_count=2, idx_header=""):
        from functools import partial
        
        if type(sequences) is str:
            sequences = [sequences]
        pbar = tqdm(total=len(sequences))

        whole_read_counts = Counter(
            [seq for experiment in self.rounds for seq in experiment.get_filter_passed_sequences(random_only=True)])
        whole_reads = [x[0] for x in filter(
            lambda x:x[1] >= min_count, whole_read_counts.most_common())]
        pbar.set_description("calculating kmer distribution")
        if not hasattr(self, "whole_reads_kmer"):
            self.whole_reads_kmer = np.stack(
                [np.array(self.kmer_count(seq, to_list=True)) for seq in whole_reads])

        with open(save_path, "w") as f:
            f.write(
                "idx\tgenerate_seq\texact_match_in\tnearest_selex_seq\tscore\tselex_aligned\tgenerate_aligned\t")
            f.write(
                "\t".join([experiment.name for experiment in self.rounds])+"\n")

            # for sequence in query
            for idx, most_probable_seq in enumerate(sequences):
                pbar.set_description("taking alignments")
                func = partial(local_alignment, s2=most_probable_seq)
                with Pool() as p:
                    alignments = p.map(func, whole_reads)
                arr = np.array(list(zip(*alignments))[0])

                matches = self.has_exact_match(most_probable_seq)[
                    most_probable_seq]

                pbar.set_description("writing to file")
                for jdx in np.argwhere(arr == max(arr)).flatten():
                    score = alignments[jdx][0]
                    opt_seq_align, check_seq_align = alignments[jdx][1][0]
                    f.write(
                        f"{idx_header}{idx}\t{most_probable_seq}\t{matches}\t{whole_reads[jdx]}\t{score}\t{opt_seq_align}\t{check_seq_align}")

                    for experiment in self.rounds:
                        c = experiment.get_filter_passed_sequences_and_count(
                            random_only=True)
                        if whole_reads[jdx] in c.keys():
                            ratio = f"\t{c[whole_reads[jdx]] / len(experiment.get_filter_passed_sequences()):.6f}"
                            f.write(ratio)
                        else:
                            f.write("\t0.000000")
                    f.write("\n")
                pbar.update(1)
        return pd.read_table(save_path)
