
from time import time

import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch import linalg
import random
import numpy as np


from baselines import RGD_QR
from proposed import RSDM

import pickle

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def procruste_loss(X):
    temp = X @ A - B
    return (temp*temp).sum()/(2*p)


def procruste_optgap(X):
    return abs(procruste_loss(X).item() - loss_star)/abs(loss_star)


def procruste_dist2ortho(X):
    temp = X.t() @ X - torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    return temp.norm()



def load_args(case):

    if case == 1:
        # THIS
        n = 2000
        p = 2000
        lowdim = 900
        method_names = ["RGD", "RSDM-P", "RSDM-O"]
        methods = [RGD_QR,  RSDM, RSDM]
        learning_rates = [0.5,  2, 2]
        n_epochs = [2000, 3000, 3000]

    return n, p, lowdim, method_names, methods, learning_rates, n_epochs



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load args
    n, p, lowdim, method_names, methods, learning_rates, n_epochs = load_args(1)

    # # create data
    A = torch.randn(p, p).to(device)
    B = torch.randn(n, p).to(device)
    init_weights = linalg.qr(torch.randn(n, p))[0]

    # Compute closed-form solution from svd, used for monitoring.
    BAt = B.matmul(A.transpose(-1, -2))
    u, _, vh = torch.linalg.svd(BAt, full_matrices=False)
    # u, _, v = torch.svd(BAt)
    w_star = u.matmul(vh)
    loss_star = procruste_loss(w_star)
    loss_star = loss_star.item()

    all_results = {}

    time_epochs_all = []
    losses_epochs_all = []
    optgap_epochs_all = []
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']
    markers = ['*', 'o', '+', 'd', '>']
    markers = [''] * 5


    tolerance = 1e-6

    if not isinstance(n_epochs, list):
        n_epochs = [n_epochs] * len(method_names)

    for method_name, method, learning_rate, n_epoch in zip(method_names, methods, learning_rates, n_epochs):

        W = nn.Parameter(torch.empty(n, p))
        W.data = init_weights.clone().to(device)

        if method_name == 'RSDM-P':
            optimizer = method([W], learning_rate, r=lowdim, use_permutation=True)
        elif method_name == 'RSDM-O':
            optimizer = method([W], learning_rate, r=lowdim, use_permutation=False)
        else:
            optimizer = method([W], learning_rate)

        # init
        loss_ = procruste_loss(W.data).cpu().item()
        dist2opt = procruste_optgap(W.data)
        losses = [loss_]
        time_epochs = [0]
        flop_epochs = [0]
        optgap_epochs = [dist2opt]
        print(
            "|%s|    time for an epoch : %.1e sec, Loss: %.2e, optgap: %.2e"
            % (method_name, time_epochs[-1], loss_, dist2opt)
        )

        counter = 0
        for epoch in range(n_epoch):
            # train
            t0 = time()

            optimizer.zero_grad()
            loss_ = procruste_loss(W)
            loss_.backward()
            optimizer.step()

            # test
            loss_ = procruste_loss(W.data).cpu().item()
            dist2opt = procruste_optgap(W.data)
            dist2ortho = procruste_dist2ortho(W.data)
            time_epochs.append(time() - t0)
            losses.append(loss_)
            optgap_epochs.append(dist2opt)
            print(
                "|%s|    time for an epoch : %.1e sec, Loss: %.2e, optgap: %.2e, dist2ortho: %.2e"
                % (method_name, time_epochs[-1], loss_, dist2opt, dist2ortho)
            )
            if dist2opt < tolerance:
                print(f"Tolerance reached at iter {epoch}, break!")
                break

        all_results[method_name] = {'time': time_epochs, 'loss': losses, 'optgap': optgap_epochs}

    with open(f'procruste_{n}_{p}_{lowdim}.pkl', 'wb') as f:
        pickle.dump(all_results, f)


    with open(f'procruste_{n}_{p}_{lowdim}.pkl', 'rb') as f:
        all_results = pickle.load(f)

    colors_map = {"RGD": "tab:blue", "RSDM-P": "tab:purple", "RSDM-O": "tab:pink"}


    plt.figure(figsize=(5.5, 4.5))
    for method_name in all_results:

        method_results = all_results[method_name]

        plt.semilogy(
            torch.cumsum(torch.tensor(method_results['time']), dim=0),
            method_results['optgap'],
            label=method_name,
            color=colors_map[method_name],
            linewidth=2.5,
        )
    plt.legend(loc=1, prop={'size': 16})
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Optimality Gap", fontsize=20)

    # plt.tight_layout()
    plt.savefig(f'st_{n}_{p}_{lowdim}.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()
