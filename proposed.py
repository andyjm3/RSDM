import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from torch import linalg



class RSDM(Optimizer):
    ''' Implementation of Riemannian submanifold descent method
    '''

    def __init__(self, params, lr, r, use_permutation=True):
        defaults = dict(lr=lr)
        super(RSDM, self).__init__(params, defaults)
        self.flops = 0
        self.use_permutation=use_permutation
        self.r = r

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        loss = None
        for p in group['params']:
            d_p = p.grad
            if self.use_permutation:

                pidx = torch.randperm(p.data.shape[0]).to(p.device)
                P = torch.eye(p.data.shape[0]).to(p.device)
                pidxr = pidx[:self.r]
                Psub = P[pidxr,:]

                PegradX = d_p.data[pidxr,:]
                PX = p.data[pidxr,:]
                gradIr = torch.mm(PegradX, PX.T)
                gradIr = (gradIr - gradIr.T)/2
                q_temp, r_temp = linalg.qr(torch.eye(self.r).to(gradIr.device) - group['lr'] * gradIr)
                unflip = torch.diagonal(r_temp).sign().add(0.5).sign()
                Y = q_temp * unflip[..., None, :]

                Xtemp = p.data[pidxr, :]
                X = p.data + Psub.T @ ( Y @ Xtemp - Xtemp )

                p.data = X
            else:
                P, _ = linalg.qr(torch.randn(p.data.shape[0],self.r, device=p.device))
                P = P.T

                PegradX = P @ d_p.data
                PX = P @ p.data
                gradIr = PegradX @ PX.T
                gradIr = (gradIr - gradIr.T) / 2
                q_temp, r_temp = linalg.qr(torch.eye(self.r).to(gradIr.device) - group['lr'] * gradIr)
                unflip = torch.diagonal(r_temp).sign().add(0.5).sign()
                Y = q_temp * unflip[..., None, :]

                X = p.data + P.T @ ( (Y - torch.eye(self.r).to(p.device)) @ (P @ p.data) )

                p.data = X

        return loss

