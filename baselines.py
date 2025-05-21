import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from torch import linalg
import numpy as np




class RGD_QR(Optimizer):
    ''' Implementation of stochastic Riemannian gradient descent with QR retraction.
    '''

    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(RGD_QR, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        loss = None
        for p in group['params']:
            d_p = p.grad
            # Riem grad
            XtG = torch.mm(torch.transpose(p.data, 0, 1), d_p.data)
            symXtG = 0.5 * (XtG + torch.transpose(XtG, 0, 1))
            Riema_grad = d_p.data - torch.mm(p.data, symXtG)
            # qr_unique
            q_temp, r_temp = linalg.qr(p.data - group['lr'] * Riema_grad)
            unflip = torch.diagonal(r_temp).sign().add(0.5).sign()
            q_temp *= unflip[..., None, :]

            p.data = q_temp
        return loss


