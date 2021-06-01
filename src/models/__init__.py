from enum import IntEnum, Enum
import math
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)


class CNNRNNVAE(nn.Module):
    class EncoderCNNRNN(nn.Module):
        def __init__(self, hidden_size, window_size=7):
            """ window size is same to the length of padding +1.
            input = N x (L + padding) x 5
            """
            super(CNNRNNVAE.EncoderCNNRNN, self).__init__()


def train(epochs, model, train_loader, test_loader, optimizer, loss_fn=None, device="cuda", n_print=100, model_str="model.mdl", save_dir=Path("./"), threshold=20, beta=1, beta_schedule=False, force_matching=False, force_epochs=20, logs=True, position=0):
    csv_filename = model_str.replace(".mdl", ".csv")
    if loss_fn == profile_hmm_loss_fn and force_matching:
        logger.info(f"force till {force_epochs}")
    patient = 0
    losses = []
    test_losses = []

    if loss_fn is None:
        loss_fn = model.loss_fn

    with tqdm(total=epochs, position=position+1) as pbar:
        description = ""
        for epoch in range(1, epochs + 1):
            if beta_schedule and epoch < threshold:
                beta = epoch / threshold
            model.train()
            train_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                if loss_fn == profile_hmm_loss_fn and epoch <= force_epochs:
                    loss = loss_fn(data, *model(data), beta=beta,
                                   force_matching=force_matching, match_cost=1+4*(1-epoch/force_epochs))
                else:
                    loss = loss_fn(data, *model(data), beta=beta)
                loss.backward()
                train_loss += loss.item() * data.shape[0]
                optimizer.step()
            train_loss /= len(train_loader.dataset)
            if np.isnan(train_loss):
                logger.info("!-- train -> nan")
                if len(losses) == 0:
                    return [(float("inf"), float("inf"), float("inf"), float("inf"))]
                return losses
            model.eval()
            test_ce = 0
            test_kld = 0
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    ce, kld = loss_fn(data, *model(data), beta=beta, test=True)
                    test_ce += ce * data.shape[0]
                    test_kld += kld * data.shape[0]
            test_kld /= len(test_loader.dataset)
            test_ce /= len(test_loader.dataset)
            test_loss = test_kld + test_ce
            if np.isnan(test_loss):
                logger.info("!-- test -> nan")
                if len(losses) == 0:
                    return [(float("inf"), float("inf"), float("inf"), float("inf"))]
                return losses
            loss = (train_loss, test_loss, test_ce, test_kld)
            losses.append(loss)
            test_losses.append(test_loss)
            if len(test_losses) - 1 == np.argmin(test_losses):
                torch.save(model.state_dict(), save_dir / model_str)
                patient = 0
            else:
                patient += 1
                if patient > threshold:
                    # logger.info(f"{epoch}: no progress in test loss for {patient} iteration. break.")
                    return losses

            patience_str = f"[{patient}]" if patient > 0 else (
                "[" + "⠸⠴⠦⠇⠋⠙"[epoch % 6] + "]")
            len_model_str = len(model_str)
            if len_model_str > 10:
                model_str_print = f"..........{model_str}.........."[
                    (epoch+9) % (len_model_str+10):(epoch+9) % (len_model_str+10) + 10]
            else:
                model_str_print = model_str
            description = f'{patience_str:>4}{epoch:4d} itr {train_loss:6.2f} <-> {test_loss:6.2f} ({test_ce:6.2f}+{test_kld:6.2f}) of {model_str_print}'

            if epoch == 1:
                with open(save_dir / csv_filename, "w") as f:
                    f.write("epoch,train_loss,test_loss,test_recon,test_kld\n")
            with open(save_dir / csv_filename, "a") as f:
                f.write(f"{epoch}," + ",".join(map(str, loss)) + "\n")
            if logs:
                logger.debug(description)
                pbar.set_description(description)
                pbar.update(1)
    return losses


def kld_loss(mu, logvar):
    KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) -
                            logvar.exp()) / mu.shape[0]
    return KLD


def ce_loss(recon_param, input):
    CE = F.cross_entropy(recon_param, input, reduction="sum") / input.shape[0]
    return CE


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


def profile_hmm_loss(recon_param, input, force_matching=False, match_cost=5):
    batch_size, random_len = input.shape

    a, e_m = recon_param
    motif_len = e_m.shape[1]

    F = torch.ones((batch_size, 3, motif_len + 1, random_len + 1),
                   device=input.device) * (-100)
    # init
    F[:, 0, 0, 0] = 0

    for i in range(random_len + 1):
        for j in range(motif_len + 1):
            # State M
            if j*i != 0:
                F[:, State.M, j, i] = e_m[:, j - 1].gather(1, input[:, i - 1:i])[:, 0] + \
                    torch.logsumexp(torch.stack((
                        a[:, j - 1, Transition.M2M] +
                        F[:, State.M, j - 1, i - 1],
                        a[:, j - 1, Transition.I2M] +
                        F[:, State.I, j - 1, i - 1],
                        a[:, j - 1, Transition.D2M] +
                        F[:, State.D, j - 1, i - 1])), dim=0)

            # State I
            if i != 0:
                F[:, State.I, j, i] = - 1.3863 + \
                    torch.logsumexp(torch.stack((
                        a[:, j, Transition.M2I] +
                        F[:, State.M, j, i-1],
                        # Removed D-to-I transition
                        # a[:, j, Transition.D2I] +
                        # F[:, State.D, j, i-1],
                        a[:, j, Transition.I2I] +
                        F[:, State.I, j, i-1]
                    )), dim=0)

            # State D
            if j != 0:
                F[:, State.D, j, i] = \
                    torch.logsumexp(torch.stack((
                        a[:, j - 1, Transition.M2D] +
                        F[:, State.M, j - 1, i],
                        # REMOVED I-to-D transition
                        # a[:, j - 1, Transition.I2D] +
                        # F[:, State.I, j - 1, i],
                        a[:, j - 1, Transition.D2D] +
                        F[:, State.D, j - 1, i]
                    )), dim=0)

    # final I->M transition
    F[:, State.M, motif_len, random_len] += a[:,
                                              motif_len, Transition.M2M]
    F[:, State.I, motif_len, random_len] += a[:,
                                              motif_len, Transition.I2M]
    F[:, State.D, motif_len, random_len] += a[:,
                                              motif_len, Transition.D2M]

    if force_matching:
        force_loss = np.log((match_cost+1)*match_cost/2) + \
            torch.sum((match_cost-1) * a[:, :, Transition.M2M], dim=1).mean()
        return - force_loss - torch.logsumexp(F[:, :, motif_len, random_len], dim=1).mean()
    return - torch.logsumexp(F[:, :, motif_len, random_len], dim=1).mean()


def torch_multi_polytope_dp_log(transition_proba, emission_proba, output):
    """
    torch_multi_polytope_dp_log(
        emission_proba,
        transition_proba,
        output
    )

    Given logarithmic parameters, the function calculates 
    the probability of the output sequence of a certain 
    Profile Hidden Markov Model (PHMM) by forward algorithm.

    For the efficiency, this function is utilizing polytope 
    model which enables parallel dynamic programming (DP).

    Parameters
    ----------
    emission_proba : torch.Tensor
        the tensor which emit characters. the tensor shape
        has to be (`batch`, `from`=3, `to`=3, `model_length`+1) and
        the tensor has to be logarithmic number

    transition_proba : torch.Tensor
        the tensor which define the probability to transit
        state to state. the tensor shape has to be:
        (`batch`, `model_length`, `augc`=4)

    output : torch.Tensor
        the tensor of the output vector. the tensor shape
        has to be (`batch`, `string_length`)

    Returns
    ------
    probabilities : torch.Tensor
        log-probabilities of the given output tensor with
        shape (`batch`,)

    """
    model_length = emission_proba.shape[1]
    batch_size, string_length = output.shape

    F = torch.ones(
        size=(batch_size, 3, model_length +
              string_length + 1, string_length + 1),
        device=output.device) * - 200
    F[:, State.M, 0, 0] = 0
    log4 = torch.Tensor([4]).log().to(output.device)
    arange = torch.arange(
        start=0, end=model_length + string_length + 1, device=output.device)
    for model_index_pre in range(1, model_length + string_length+1):
        if max(1, model_index_pre-model_length) < min(string_length+1, model_index_pre):
            m_slice = arange[max(1, model_index_pre-model_length)                             : min(string_length+1, model_index_pre)]
            F[:, State.M, model_index_pre, m_slice] = \
                torch.gather(
                    emission_proba[:, model_index_pre - m_slice - 1], 2,
                    output[:, m_slice - 1, None]).reshape(batch_size, len(m_slice)) \
                + torch.logsumexp(
                    transition_proba[:, :, State.M,
                                     model_index_pre - m_slice - 1]
                    + F[:, :, model_index_pre - 2, m_slice - 1], axis=1)

        if max(1, model_index_pre-model_length) < min(string_length+1, model_index_pre+1):
            i_slice = arange[max(1, model_index_pre-model_length)                             : min(string_length+1, model_index_pre+1)]
            F[:, State.I, model_index_pre, i_slice] = \
                torch.logsumexp(
                    transition_proba[:, :, State.I, model_index_pre - i_slice]
                    + F[:, :, model_index_pre - 1, i_slice - 1], axis=1) \
                - log4

        if max(0, model_index_pre-model_length) < min(string_length+1, model_index_pre):
            d_slice = arange[max(1, model_index_pre-model_length)                             : min(string_length+1, model_index_pre)]
            F[:, State.D, model_index_pre, d_slice] = \
                torch.logsumexp(
                    transition_proba[:, :, State.D,
                                     model_index_pre - d_slice - 1]
                    + F[:, :, model_index_pre - 1, d_slice], axis=1)

    return -torch.logsumexp(F[:, :, -1, -1] + transition_proba[:, :, State.M, -1], axis=1).mean()


def end_padded_multi_categorical_loss_fn(input, recon_param, mu, logvar, debug=False, test=False, beta=1):
    from src.data import nt_index
    loss = multi_categorical_loss_fn(
        F.pad(input, (0, 1), "constant", nt_index.EOS),
        recon_param, mu, logvar, debug, test, beta)
    # logger.info(loss.shape)
    return loss


def multi_categorical_loss_fn(input, recon_param, mu, logvar, debug=False, test=False, beta=1):
    ce = ce_loss(recon_param, input)
    kld = kld_loss(mu, logvar)

    if debug:
        logger.info(f"ce={ce:.2f}, kld={kld:.2f}")
    if test:
        return ce.item(), kld.item()
    return ce + beta * kld


def profile_hmm_loss_fn(input, recon_param, mu, logvar, debug=False, test=False, beta=1, force_matching=False, match_cost=5):
    phmmloss = profile_hmm_loss(
        recon_param, input, force_matching=force_matching, match_cost=match_cost)
    kld = kld_loss(mu, logvar)

    if debug:
        logger.info(f"phmm={phmmloss:.2f}, kld={kld:.2f}")
    if test:
        return phmmloss.item(), kld.item()
    return phmmloss + beta * kld


def profile_hmm_loss_fn_fast(input, recon_param, mu, logvar, debug=False, test=False, beta=1, force_matching=False, match_cost=5):
    phmmloss = torch_multi_polytope_dp_log(*recon_param, input)
    kld = kld_loss(mu, logvar)

    if debug:
        logger.info(f"phmm={phmmloss:.2f}, kld={kld:.2f}")
    if test:
        return phmmloss.item(), kld.item()
    return phmmloss + beta * kld


class SAM(nn.Module):
    def __init__(self, hidden_size=32, kernel_size=7):
        super(SAM, self).__init__()
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size

        rel_size = hidden_size//16
        intermid_size = hidden_size//8
        out_size = hidden_size//4

        concat_size = (1 + kernel_size) * rel_size

        linear_kwargs = {
            "in_channels": hidden_size,
            "out_channels": rel_size,
            "kernel_size": 1}

        calc_weight_intermid_kwargs = {
            "in_channels": concat_size,
            "out_channels": intermid_size,
            "kernel_size": 1}

        calc_weight_kwargs = {
            "in_channels": intermid_size,
            "out_channels": out_size,
            "kernel_size": 1}

        beta_kwargs = {
            "in_channels": hidden_size,
            "out_channels": out_size,
            "kernel_size": 1}

        out_kwargs = {
            "in_channels": out_size,
            "out_channels": hidden_size,
            "kernel_size": 1}

        self.x2phi = nn.Conv1d(**linear_kwargs)
        self.x2psi = nn.Conv1d(**linear_kwargs)
        self.x2beta = nn.Conv1d(**beta_kwargs)
        self.aggr2out = nn.Conv1d(**out_kwargs)

        self.calc_weight = nn.Sequential(
            nn.BatchNorm1d(concat_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(**calc_weight_intermid_kwargs),
            nn.BatchNorm1d(intermid_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(**calc_weight_kwargs)
        )

        self.bn_aggr = nn.BatchNorm1d(out_size)
        self.bn_out = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        # xは (batch_size, Channel, Length) のvector
        phi = self.x2phi(x)
        psi = self.x2psi(x)

        # since unfold only support 4d input
        psi = torch.flatten(F.unfold(
            input=psi[:, :, :, None],
            kernel_size=(self.kernel_size, 1),
            padding=(self.kernel_size//2, 0)), start_dim=2)

        weight = self.calc_weight(
            torch.cat((phi, psi), dim=1))

        beta = self.x2beta(x)

        aggr = F.relu(self.bn_aggr(weight * beta))
        return F.relu(self.bn_out(x + self.aggr2out(aggr)))


class Bottleneck(nn.Module):
    def __init__(self, init_dim=32, window_size=7):
        super(Bottleneck, self).__init__()
        assert window_size % 2 == 1, f"window size should be odd, given {window_size}"

        self.conv1 = nn.Conv1d(
            in_channels=init_dim,
            out_channels=init_dim*2,
            kernel_size=1)

        self.conv2 = nn.Conv1d(
            in_channels=init_dim*2,
            out_channels=init_dim*2,
            kernel_size=window_size,
            padding=window_size//2
        )

        self.conv3 = nn.Conv1d(
            in_channels=init_dim*2,
            out_channels=init_dim,
            kernel_size=1)

        self.bn1 = nn.BatchNorm1d(init_dim)
        self.bn2 = nn.BatchNorm1d(init_dim*2)
        self.bn3 = nn.BatchNorm1d(init_dim*2)

    def forward(self, input):
        x = self.conv1(F.leaky_relu(self.bn1(input)))
        x = self.conv2(F.leaky_relu(self.bn2(x)))
        x = self.conv3(F.leaky_relu(self.bn3(x)))
        return F.leaky_relu(x+input)


class EncoderCNN (nn.Module):
    # 0~3 is already used by embedding ATGC
    def __init__(self, embedding_dim=32, window_size=7, num_layers=6):
        super(EncoderCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size

        self.embed = nn.Embedding(
            num_embeddings=4,  # [A,T,G,C,PAD,SOS,EOS]
            embedding_dim=embedding_dim)

        modules = [Bottleneck(embedding_dim, window_size)
                   for _ in range(num_layers)]
        self.resnet = nn.Sequential(*modules)

    def forward(self, seqences):
        # change X from (N, L) to (N, L, C)
        x = F.leaky_relu(self.embed(seqences))

        # change X to (N, C, L)
        x = x.transpose(1, 2)
        value, indices = self.resnet(x).max(dim=2)
        return value


class EncoderLSTM (nn.Module):
    # 0~3 is already used by embedding ATGC
    def __init__(self, embedding_dim=32, window_size=7):
        super(EncoderLSTM, self).__init__()
        from src.data import nt_index
        self.embedding_dim = embedding_dim
        self.window_size = window_size

        self.embed = nn.Embedding(
            num_embeddings=7,  # [A,T,G,C,PAD,SOS,EOS]
            embedding_dim=embedding_dim,
            padding_idx=int(nt_index.PAD))

        self.lstm = nn.LSTM(embedding_dim,
                            embedding_dim//2,
                            bidirectional=True)

    def forward(self, seqences):
        from src.data import nt_index

        # PAD, SOS, SEQ, EOS, PAD
        x = F.pad(seqences, (1, 0), "constant", int(nt_index.SOS))
        x = F.pad(x, (0, 1), "constant", int(nt_index.EOS))

        x = self.embed(x)
        x = x.transpose(0, 1)
        o, (h, c) = self.lstm(x)

        return torch.cat((h[0], h[1]), dim=1)


class EncoderCNNLSTM (nn.Module):
    # 0~3 is already used by embedding ATGC
    def __init__(self, embedding_dim=32, window_size=7):
        super(EncoderCNNLSTM, self).__init__()
        from src.data import nt_index
        self.embedding_dim = embedding_dim
        self.window_size = window_size

        self.embed = nn.Embedding(
            num_embeddings=7,  # [A,T,G,C,PAD,SOS,EOS]
            embedding_dim=embedding_dim,
            padding_idx=int(nt_index.PAD))

        self.lstm = nn.LSTM(embedding_dim,
                            embedding_dim//2,
                            bidirectional=True)

        self.resnet = nn.Sequential(
            Bottleneck(embedding_dim, window_size),
            Bottleneck(embedding_dim, window_size))

    def forward(self, seqences):
        from src.data import nt_index
        # PAD, SOS, SEQ, EOS, PAD
        x = F.pad(seqences, (1, 0), "constant", int(nt_index.SOS))
        x = F.pad(x, (0, 1), "constant", int(nt_index.EOS))
        x = F.pad(x, (self.window_size-1, self.window_size-1),
                  "constant", int(nt_index.PAD))

        # change X from (N, L) to (N, L, C)
        x = F.leaky_relu(self.embed(x))

        # change X to (N, C, L)
        x = x.transpose(1, 2)
        x = self.resnet(x)

        x = x.permute(2, 0, 1)
        o, (h, c) = self.lstm(x)

        return torch.cat((h[0], h[1]), dim=1)


class DecoderPHMM(nn.Module):
    # tile hidden and input to make x
    def __init__(self,  motif_len, embed_size,  hidden_size=32):
        super(DecoderPHMM, self).__init__()

        class View(nn.Module):
            def __init__(self, shape):
                super(View, self).__init__()
                self.shape = shape

            def forward(self, x):
                return x.view(*self.shape)

        self.fc1 = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.BatchNorm1d(hidden_size*2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size*2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.tr_from_M = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, (motif_len+1)*3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, motif_len+1, 3)),
            nn.LogSoftmax(dim=2)
        )
        self.tr_from_I = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, (motif_len+1)*2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, motif_len+1, 2)),
            nn.LogSoftmax(dim=2)
        )
        self.tr_from_D = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, (motif_len+1)*2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, motif_len+1, 2)),
            nn.LogSoftmax(dim=2)
        )

        self.emission = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, motif_len*4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, motif_len, 4)),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, input):
        x = self.fc1(input)

        transition_from_match = self.tr_from_M(x)
        transition_from_insertion = self.tr_from_I(x)
        transition_from_deletion = self.tr_from_D(x)

        emission_proba = self.emission(x)
        return (torch.cat((
            transition_from_match,
            transition_from_insertion,
            transition_from_deletion), dim=2), emission_proba)


class DecoderPHMM_fast(nn.Module):
    # tile hidden and input to make x
    def __init__(self,  motif_len, embed_size,  hidden_size=32):
        super(DecoderPHMM_fast, self).__init__()

        class View(nn.Module):
            def __init__(self, shape):
                super(View, self).__init__()
                self.shape = shape

            def forward(self, x):
                return x.view(*self.shape)

        self.fc = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))

        self.transition = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, 3*3*(motif_len+1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, 3, 3, motif_len+1)),
            nn.LogSoftmax(dim=2)
        )

        self.emission = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, motif_len*4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, motif_len, 4)),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, input):
        x = self.fc(input)

        transition_proba = self.transition(x)
        emission_proba = self.emission(x)

        return (transition_proba, emission_proba)


class DecoderCNN(nn.Module):
    # tile hidden and input to make x
    def __init__(self, embed_size, target_len, hidden_size=32, kernel_size=7, in_channels=4):
        assert kernel_size % 2 == 1, f"kernel_size has to be odd. given k={kernel_size}."
        super(DecoderCNN, self).__init__()

        class View(nn.Module):
            def __init__(self, shape):
                super(View, self).__init__()
                self.shape = shape

            def forward(self, x):
                return x.view(*self.shape)

        transpose_kwargs = {
            "in_channels": hidden_size,
            "out_channels": hidden_size,
            "kernel_size": kernel_size,
            "padding": kernel_size//2,
            "stride": 1}
        conv1x1_kwargs = {
            "in_channels": hidden_size,
            "out_channels": in_channels,
            "kernel_size": kernel_size,
            "padding": kernel_size//2,
            "stride": 1}
        batchnorm_kwargs = {
            "num_features": hidden_size,
            "eps": 1e-5,
            "momentum": 0.1,
            "affine": True
        }

        self.fc1 = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.BatchNorm1d(hidden_size*2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size*2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(negative_slope=0.01))

        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * target_len),
            View(shape=(-1, hidden_size, target_len)),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose1d(**transpose_kwargs),

            nn.BatchNorm1d(**batchnorm_kwargs),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose1d(**transpose_kwargs),

            nn.BatchNorm1d(**batchnorm_kwargs),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose1d(**conv1x1_kwargs),

            nn.LeakyReLU(negative_slope=0.01, inplace=True))

    def forward(self, in_x):
        in_x = self.fc1(in_x)
        x = self.fc2(in_x)
        x += in_x
        return self.network(x)


class DecoderRNN (nn.Module):
    # 0~3 is already used by embedding ATGC
    def __init__(self, embed_size, hidden_size, teacher_forcing_ratio):
        super(DecoderRNN, self).__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.fc1 = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(negative_slope=0.01))

        self.embed = nn.Embedding(
            num_embeddings=7,
            embedding_dim=hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size)

        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size, 7),
            nn.LeakyReLU(negative_slope=0.01))
        from src.data import nt_index
        self.nt_index = nt_index

    def forward(self, in_x: torch.Tensor, seqences: torch.Tensor) -> torch.Tensor:

        in_x = self.fc1(in_x)
        hidden = self.fc2(in_x)
        hidden += in_x
        hidden = hidden.unsqueeze(0)

        if torch.rand([1]).item() < self.teacher_forcing_ratio:
            # teacher forcingに使用するinput
            x = F.pad(seqences, (1, 0), "constant", self.nt_index.SOS)
            x = self.embed(x)
            x = x.transpose(0, 1)
            o, h = self.gru(x, hidden)
            return F.leaky_relu(self.out(o)).permute(1, 2, 0)
        else:
            x = torch.ones(
                hidden.shape[1], 1, dtype=torch.long, device=hidden.device) * self.nt_index.SOS
            x = self.embed(x)
            x = x.transpose(0, 1)

            outputs = []
            for _ in range(seqences.shape[1]+1):
                output, hidden = self.gru(x, hidden)
                output = self.out(output)
                outputs.append(output)
                # for next input
                topv, topi = output.topk(1)
                x = self.embed(topi.view(1, -1).detach())
            return F.leaky_relu(torch.cat(outputs, dim=0)).permute(1, 2, 0)


class VAE(nn.Module):
    def __init__(self, encoder, decoder, embed_size=10, hidden_size=32):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.h2mu = nn.Linear(hidden_size, embed_size)
        self.h2logvar = nn.Linear(hidden_size, embed_size)

    def reparameterize(self, mu, logvar, deterministic=False):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + (std * eps if not deterministic else 0)
        return z

    def forward(self, input, deterministic=False):
        h = self.encoder(input)
        mu = self.h2mu(h)
        logvar = self.h2logvar(h)

        z = self.reparameterize(mu, logvar, deterministic)
        recon_param = self.decoder(z)
        return recon_param, mu, logvar


class CNN_Mul_VAE(VAE):
    def __init__(self, target_len, embed_size=10, hidden_size=32, kernel_size=13):
        encoder = EncoderCNN(hidden_size, kernel_size)
        decoder = DecoderCNN(embed_size, target_len)
        super(CNN_Mul_VAE, self).__init__(
            encoder, decoder, embed_size, hidden_size)

        self.loss_fn = multi_categorical_loss_fn


class LSTM_Mul_VAE(VAE):
    def __init__(self, target_len, embed_size=10, hidden_size=32, kernel_size=13):
        encoder = EncoderLSTM(hidden_size, kernel_size)
        decoder = DecoderCNN(embed_size, target_len)
        super(LSTM_Mul_VAE, self).__init__(
            encoder, decoder, embed_size, hidden_size)

        self.loss_fn = multi_categorical_loss_fn


class CNNLSTM_Mul_VAE(VAE):
    def __init__(self, target_len, embed_size=10, hidden_size=32, kernel_size=13):
        encoder = EncoderCNNLSTM(hidden_size, kernel_size)
        decoder = DecoderCNN(embed_size, target_len)
        super(CNNLSTM_Mul_VAE, self).__init__(
            encoder, decoder, embed_size, hidden_size)

        self.loss_fn = multi_categorical_loss_fn


class CNN_PHMM_VAE(VAE):
    def __init__(self, motif_len=12, embed_size=10, hidden_size=32, kernel_size=7):
        encoder = EncoderCNN(hidden_size, kernel_size)
        decoder = DecoderPHMM(motif_len, embed_size)

        super(CNN_PHMM_VAE, self).__init__(
            encoder, decoder, embed_size, hidden_size)
        self.loss_fn = profile_hmm_loss_fn


class CNN_PHMM_VAE_FAST(VAE):
    def __init__(self, motif_len=12, embed_size=10, hidden_size=32, kernel_size=7):
        encoder = EncoderCNN(hidden_size, kernel_size)
        decoder = DecoderPHMM_fast(
            motif_len, embed_size, hidden_size=hidden_size)

        super(CNN_PHMM_VAE_FAST, self).__init__(
            encoder, decoder, embed_size, hidden_size)
        self.loss_fn = profile_hmm_loss_fn_fast


class LSTM_PHMM_VAE(VAE):
    def __init__(self, motif_len=12, embed_size=10, hidden_size=32, kernel_size=7):
        encoder = EncoderLSTM(hidden_size, kernel_size)
        decoder = DecoderPHMM(motif_len, embed_size)
        super(LSTM_PHMM_VAE, self).__init__(
            encoder, decoder, embed_size, hidden_size)

        self.loss_fn = profile_hmm_loss_fn


class CNNLSTM_PHMM_VAE(VAE):
    def __init__(self, motif_len=12, embed_size=10, hidden_size=32, kernel_size=7):
        encoder = EncoderCNNLSTM(hidden_size, kernel_size)
        decoder = DecoderPHMM(motif_len, embed_size)
        super(CNNLSTM_PHMM_VAE, self).__init__(
            encoder, decoder, embed_size, hidden_size)

        self.loss_fn = profile_hmm_loss_fn


class CNN_AR_VAE(VAE):
    def __init__(self, embed_size=10, hidden_size=32, kernel_size=13, teacher_forcing_ratio=1):
        encoder = EncoderCNN(hidden_size, kernel_size)
        decoder = DecoderRNN(embed_size, hidden_size, teacher_forcing_ratio)
        super(CNN_AR_VAE, self).__init__(
            encoder, decoder, embed_size, hidden_size)

        self.loss_fn = end_padded_multi_categorical_loss_fn

    def forward(self, input, deterministic=False):
        h = self.encoder(input)
        mu = self.h2mu(h)
        logvar = self.h2logvar(h)

        z = self.reparameterize(mu, logvar, deterministic)
        recon_param = self.decoder(z, input)
        return recon_param, mu, logvar


class LSTM_AR_VAE(VAE):
    def __init__(self, embed_size=10, hidden_size=32, kernel_size=13,  teacher_forcing_ratio=1):
        encoder = EncoderLSTM(hidden_size, kernel_size)
        decoder = DecoderRNN(embed_size, hidden_size, teacher_forcing_ratio)
        super(LSTM_AR_VAE, self).__init__(
            encoder, decoder, embed_size, hidden_size)

        self.loss_fn = end_padded_multi_categorical_loss_fn

    def forward(self, input, deterministic=False):
        h = self.encoder(input)
        mu = self.h2mu(h)
        logvar = self.h2logvar(h)

        z = self.reparameterize(mu, logvar, deterministic)
        recon_param = self.decoder(z, input)
        return recon_param, mu, logvar


class CNNLSTM_AR_VAE(VAE):
    def __init__(self, embed_size=10, hidden_size=32, kernel_size=13, teacher_forcing_ratio=1):
        encoder = EncoderCNNLSTM(hidden_size, kernel_size)
        decoder = DecoderRNN(embed_size, hidden_size, teacher_forcing_ratio)
        super(CNNLSTM_AR_VAE, self).__init__(
            encoder, decoder, embed_size, hidden_size)

        self.loss_fn = end_padded_multi_categorical_loss_fn

    def forward(self, input, deterministic=False):
        h = self.encoder(input)
        mu = self.h2mu(h)
        logvar = self.h2logvar(h)

        z = self.reparameterize(mu, logvar, deterministic)
        recon_param = self.decoder(z, input)
        return recon_param, mu, logvar
