# https://repository.rice.edu/server/api/core/bitstreams/3d52e0a2-b1c6-45ce-84f7-42774b933099/content

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



def loss_(X):
    return torch.trace(A.t() @ (X * X) @ B @ (X*X).t())/(2)

def optgap_(X):
    return abs(loss_(X).item() - loss_star)/abs(loss_star)


def load_args(case):
    if case == 1:
        n = 1000
        p = 1000
        lowdim = 500
        method_names = ["RGD", "RSDM-P", "RSDM-O"]
        methods = [RGD_QR, RSDM, RSDM]
        learning_rates = [0.005, 0.005, 0.01]
        n_epochs = [3500, 3500, 3500]
        seed = 42
        loss_star = -38464.5

    return n, p, lowdim, method_names, methods, learning_rates, n_epochs, seed, loss_star

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    n, p, lowdim, method_names, methods, learning_rates, n_epochs, seed, loss_star = load_args(1)

    seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    A = torch.randn(n,n,device=device)
    B = torch.randn(n,n,device=device)
    init_weights = linalg.qr(torch.randn(n, p))[0]


    all_results = {}
    numupate = 100
    time_epochs_all = []
    losses_epochs_all = []
    flop_epochs_all = []
    optgap_epochs_all = []
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']
    markers = ['*', 'o', '+', 'd', '>']
    markers = [''] * 5
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
        loss = loss_(W.data).cpu().item()
        dist2opt = optgap_(W.data)
        losses = [loss]
        time_epochs = [0]
        flop_epochs = [0]
        optgap_epochs = [dist2opt]
        print(
            "|%s|    time for an epoch : %.1e sec, Loss: %.2e"
            % (method_name, time_epochs[-1], loss)
        )

        for epoch in range(n_epoch):
            t0 = time()

            optimizer.zero_grad()
            loss = loss_(W)
            loss.backward()
            optimizer.step()

            loss = loss_(W.data).cpu().item()
            dist2opt = optgap_(W.data)
            time_epochs.append(time() - t0)
            losses.append(loss)
            optgap_epochs.append(dist2opt)
            print(
                "|%s|    time for an epoch : %.1e sec, Loss: %.2e, optgap: %.2e"
                % (method_name, time_epochs[-1], loss, dist2opt)
            )

        all_results[method_name] = {'time': time_epochs, 'loss': losses, 'optgap': optgap_epochs}

    with open(f'quad_{n}_{p}_{lowdim}.pkl', 'wb') as f:
        pickle.dump(all_results, f)


    with open(f'quad_{n}_{p}_{lowdim}.pkl', 'rb') as f:
        all_results = pickle.load(f)

    num_method = len(method_names)

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
    plt.xlim([0, 36])
    plt.ylim([2E-3, 2])
    plt.tight_layout()
    plt.savefig(f'quad_{n}_{p}_{lowdim}.pdf', bbox_inches='tight')
    plt.close()

