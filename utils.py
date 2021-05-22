import os
import torch
import random
import numpy as np
from PIL import Image

def fix_seeds(seed=666):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def clear_line():
    print(' ' * 30, end='\r')

def fgsm(model, x, y, criterion, epsilon):
    delta = torch.zeros_like(x, requires_grad=True)
    loss = criterion(model(x+delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd(model, x, y, criterion, epsilon, alpha, num_iter=20):
    delta = torch.zeros_like(x, requires_grad=True)
    for i in range(num_iter):
        loss = criterion(model(x+delta), y)
        loss.backward()
        d = (delta + alpha*delta.grad.detach().sign())
        delta.data = torch.max(torch.min(d, epsilon), -epsilon)
#        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()

def calc_linf(args, adv_names):
    linf = 0.0
    for name in adv_names:
        ben_im = np.array(Image.open(os.path.join(args.data_dir, name)), dtype=np.float)
        adv_im = np.array(Image.open(os.path.join(args.adv_dir, name)), dtype=np.float)
        linf = max(linf, np.max(np.abs(ben_im-adv_im)).item())
    return linf
