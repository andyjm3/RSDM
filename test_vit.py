import os
import time

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

from proposed import RSDM
from baselines import RGD_QR

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

import math
from loguru import logger

import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



class SelfAttentionOrthogonal(nn.Module):
    def __init__(self, emb_dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert emb_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # Learnable weight matrices for Q, K, V (like Linear layer, but explicitly defined)
        self.W_q_ortho = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.W_k_ortho = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.W_v_ortho = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.o_proj = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.W_q_ortho)
        nn.init.orthogonal_(self.W_k_ortho)
        nn.init.orthogonal_(self.W_v_ortho)

    def forward(self, x):
        B, N, D = x.shape  # (batch, tokens, emb_dim)

        # Manual linear projection: (B, N, D) @ (D, D) = (B, N, D)
        q = x @ self.W_q_ortho
        k = x @ self.W_k_ortho
        v = x @ self.W_v_ortho

        # Reshape for multi-head: (B, N, H, D_head) → (B, H, N, D_head)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N, N)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, H, N, D_head)
        out = out.transpose(1, 2).reshape(B, N, D)  # (B, N, D)

        # Final linear projection using W_o
        return self.o_proj(out)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, emb_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, emb_dim, H/patch, W/patch) -> flatten to (B, n_patches, emb_dim)
        x = self.proj(x)  # (B, emb_dim, n_h, n_w)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, emb_dim)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim=768, num_heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = SelfAttentionOrthogonal(emb_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=10,
                 emb_dim=768, depth=6, num_heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, emb_dim))
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.Sequential(
            *[TransformerEncoderBlock(emb_dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, n_patches, emb_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + n_patches, emb_dim)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.encoder(x)
        x = self.norm(x)
        cls_output = x[:, 0]  # extract cls token
        return self.head(cls_output)





def evaluate():
    model.eval()
    num_correct = 0
    num_samples = 0
    test_loss = 0
    counter = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            _, predictions = torch.max(scores, 1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

            test_loss += criterion(scores, targets)
            counter += 1

        acc = float(num_correct) / float(num_samples) * 100
        test_loss = test_loss /counter

    return acc, test_loss



if __name__ == '__main__':

    DATASET = 'cifar'
    RANDOM_SEED = 42
    LEARNING_RATE = 0.1
    BATCH_SIZE = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    if DATASET == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])

        train_dataset = datasets.MNIST(root='data',
                                       train=True,
                                       transform=transform,
                                       download=True)

        test_dataset = datasets.MNIST(root='data',
                                      train=False,
                                      transform=transform)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
    elif DATASET == 'cifar':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            # transforms.Lambda(lambda x: x.view(-1))
        ])

        train_dataset = datasets.CIFAR10(root='data',
                                       train=True,
                                       transform=transform,
                                       download=True)

        test_dataset = datasets.CIFAR10(root='data',
                                      train=False,
                                      transform=transform)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)



    loss_all = []
    acc_all = []
    time_all = []
    test_loss_all = []

    model = VisionTransformer(
            img_size=32,  # CIFAR-10 image size
            patch_size=4,  # 8x8 = 64 patches
            in_channels=3,
            num_classes=10,  # CIFAR-10
            emb_dim=1024,  # Smaller embedding
            depth=6,  # Fewer layers
            num_heads=4,
            mlp_dim=256,
            dropout=0.1
        ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable_params}")

    method = 'RSDM-P'

    if method == 'RGD':
        lr = 0.05
        lowdim = 300
        NUM_EPOCHS = 5
    elif method == 'RSDM-P':
        lr = 0.5
        lowdim = 300
        NUM_EPOCHS = 6

    num_rep = 5

    for rep in range(num_rep):
        loss_rep = []
        acc_rep = []
        time_rep = []
        test_loss_rep = []

        model = VisionTransformer(
            img_size=32,  # CIFAR-10 image size
            patch_size=4,  # 8x8 = 64 patches
            in_channels=3,
            num_classes=10,  # CIFAR-10
            emb_dim=1024,  # Smaller embedding
            depth=6,  # Fewer layers
            num_heads=4,
            mlp_dim=256,
            dropout=0.1
        ).to(device)


        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()

        stiefel_params = []
        other_params = []

        for name, param in model.named_parameters():
            if 'ortho' in name:
                stiefel_params.append(param)
            else:
                other_params.append(param)

        logger.info(f"method = {method}, lr = {lr}, lowdim = {lowdim}")

        if method == 'RSDM-P':
            optimizer_stiefel = RSDM(stiefel_params, lr=lr, r=lowdim, use_permutation=True)
        elif method == 'RGD':
            optimizer_stiefel = RGD_QR(stiefel_params, lr=lr) #[0.01]
        optimizer_other = torch.optim.Adam(other_params, lr=5e-4)

        start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            model.train()
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(device)
                targets = targets.to(device)

                scores = model(data)
                loss = criterion(scores, targets)

                optimizer_stiefel.zero_grad()
                optimizer_other.zero_grad()
                loss.backward()

                optimizer_stiefel.step()
                optimizer_other.step()

                if batch_idx % 100 == 0:
                    elapsed_time = time.time() - start_time
                    acc, test_loss = evaluate()
                    logger.info(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Test ACC: {acc:.2f}%, '
                          f'Time: {elapsed_time:.4f}')
                    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Test ACC: {acc:.2f}%, '
                          f'Time: {elapsed_time:.4f}')

                    loss_rep.append(loss.item())
                    time_rep.append(elapsed_time)
                    acc_rep.append(acc)
                    test_loss_rep.append(test_loss)

                    start_time = time.time()

        loss_all.append(loss_rep)
        time_all.append(time_rep)
        acc_all.append(acc_rep)
        test_loss_all.append(test_loss_rep)

        arrays = {'loss': loss_all, 'time': time_all, 'acc': acc_all, 'test_loss': test_loss_all }

    with open(f'{method}_vit_{DATASET}.pkl', 'wb') as f:
        pickle.dump(arrays, f)


    ##
    #
    with open(f'RSDM-P_vit_{DATASET}.pkl', 'rb') as f:
        RSDM_results = pickle.load(f)
        time_rsdm = RSDM_results['time']
        loss_rsdm = RSDM_results['loss']
        acc_rsdm = RSDM_results['acc']

        loss_rsdm_mean = np.array(loss_rsdm).mean(axis=0)
        loss_rsdm_std = np.array(loss_rsdm).std(axis=0)
        time_rsdm_mean = np.array(time_rsdm).mean(axis=0)
        acc_rsdm_mean = np.array(acc_rsdm).mean(axis=0)
        acc_rsdm_std = np.array(acc_rsdm).std(axis=0)

    with open(f'RGD_vit_{DATASET}.pkl', 'rb') as f:
        RGD_results = pickle.load(f)
        time_rgd = RGD_results['time']
        loss_rgd = RGD_results['loss']
        acc_rgd = RGD_results['acc']

        loss_rgd_mean = np.array(loss_rgd).mean(axis=0)
        loss_rgd_std = np.array(loss_rgd).std(axis=0)
        time_rgd_mean = np.array(time_rgd).mean(axis=0)
        acc_rgd_mean = np.array(acc_rgd).mean(axis=0)
        acc_rgd_std = np.array(acc_rgd).std(axis=0)

    print(acc_rsdm_mean)

    #
    #
    # # time
    plt.figure(figsize=(5.5, 4.5))
    plt.plot(time_rgd_mean.cumsum(), acc_rgd_mean, label='RGD', color='tab:blue')
    plt.fill_between(time_rgd_mean.cumsum(), acc_rgd_mean - acc_rgd_std,
                     acc_rgd_mean + acc_rgd_std,
                     color='tab:blue', alpha=0.3)
    plt.plot(time_rsdm_mean.cumsum(), acc_rsdm_mean, label='RSDM-P', color='tab:purple')
    plt.fill_between(time_rsdm_mean.cumsum(), acc_rsdm_mean - acc_rsdm_std,
                     acc_rsdm_mean + acc_rsdm_std,
                     color='tab:purple', alpha=0.3)
    plt.legend(loc=7, prop={'size': 16}, bbox_to_anchor=(1, 0.7))
    # plt.legend(loc='best')
    plt.xticks(fontsize=13)
    plt.xlim([0, 600])
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("ACC (%)", fontsize=20)
    plt.savefig(f'vit_acc_{DATASET}.pdf', bbox_inches='tight', dpi=300)
    plt.close()





