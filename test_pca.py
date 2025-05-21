
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



def pca_loss(X):
    return -torch.trace(X.transpose(-1,-2) @ A @ X)/(2)

def pca_optgap(X):
    return abs(pca_loss(X).item() - loss_star)/abs(loss_star)



def load_args(case):
    if case == 1:
        n = 2000
        p = 1500
        lowdim = 700
        method_names = ["RGD", "RSDM-P", "RSDM-O"]
        methods = [RGD_QR, RSDM, RSDM]
        learning_rates = [0.1, 1.5, 1.5]
        n_epochs = [1000, 1000, 1000]

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

    n, p, lowdim, method_names, methods, learning_rates, n_epochs = load_args(1)

    CN = 1000
    D = 10 * torch.diag(torch.logspace(-np.log10(CN), 0, n))
    [Q, R] = linalg.qr(torch.randn(n,n))
    A = Q @ D @ Q
    A = (A + A.t())/2

    A = A.to(device)
    init_weights = linalg.qr(torch.randn(n, p))[0]

    # Compute closed-form solution from svd, used for monitoring.
    [_, w_star] = torch.linalg.eigh(A/(2))
    w_star = w_star[:,-p:]
    loss_star = pca_loss(w_star)
    loss_star = loss_star.item()

    all_results = {}

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
        loss = pca_loss(W.data).cpu().item()
        dist2opt = pca_optgap(W.data)
        losses = [loss]
        time_epochs = [0]
        optgap_epochs = [dist2opt]
        print(
            "|%s|    time for an epoch : %.1e sec, Loss: %.2e, optgap: %.2e"
            % (method_name, time_epochs[-1], loss, dist2opt)
        )

        for epoch in range(n_epoch):
            t0 = time()

            optimizer.zero_grad()
            loss = pca_loss(W)
            loss.backward()
            optimizer.step()

            loss = pca_loss(W.data).cpu().item()
            dist2opt = pca_optgap(W.data)
            time_epochs.append(time() - t0)
            losses.append(loss)
            optgap_epochs.append(dist2opt)
            print(
                "|%s|    time for an epoch : %.1e sec, Loss: %.2e, optgap: %.2e"
                % (method_name, time_epochs[-1], loss, dist2opt)
            )

            if dist2opt < 1e-6:
                print(f"Tolerance reached. Break at iter {epoch}!")
                break

        all_results[method_name] = {'time': time_epochs, 'loss': losses, 'optgap': optgap_epochs}

        print(torch.cumsum(torch.tensor(time_epochs), dim=0)[-1])

    with open(f'pca_{n}_{p}_{lowdim}.pkl', 'wb') as f:
        pickle.dump(all_results, f)


    with open(f'pca_{n}_{p}_{lowdim}.pkl', 'rb') as f:
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
    plt.tight_layout()
    plt.savefig(f'pca_{n}_{p}_{lowdim}.pdf', bbox_inches='tight')
    plt.close()