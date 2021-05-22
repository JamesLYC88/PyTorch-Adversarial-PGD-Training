from data import AdvDataset
from utils import fix_seeds, calc_linf
from pytorchcv.model_provider import get_model as ptcv_get_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def epoch(model, loader, criterion):
    model.eval()
    train_acc, train_loss = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yp = model(x)
        loss = criterion(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
    return train_acc / len(loader.dataset), train_loss / len(loader.dataset)

def main(args):

    # pre
    fix_seeds()

    # data
    ben_set = AdvDataset(args.data_dir)
    adv_names = ben_set.__getname__()
    ben_loader = DataLoader(ben_set, batch_size=args.bs, shuffle=False)
    adv_set = AdvDataset(args.adv_dir)
    adv_loader = DataLoader(adv_set, batch_size=args.bs, shuffle=False)

    # model / criterion
    model = ptcv_get_model(args.model, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.load))
    criterion = nn.CrossEntropyLoss()

    # test
    ben_acc, _ = epoch(model, ben_loader, criterion)
    adv_acc, _ = epoch(model, adv_loader, criterion)
    linf = calc_linf(args, adv_names)
    print(f'benign accuracy = {ben_acc:.5f}')
    print(f'adversarial accuracy = {adv_acc:.5f}')
    print(f'linf = {linf:.5f}')

if __name__ == '__main__':

    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bs', type=int, default=8, help='data: batch size')
    parser.add_argument('--model', type=str, default='wrn16_10_cifar10', help='model')
    parser.add_argument('--load', type=str, default='./wrn16.pth', help='model: checkpoint')
    parser.add_argument('--data_dir', type=str, default='./benign', help='test: benign images')
    parser.add_argument('--adv_dir', type=str, default='./adv', help='test: adversarial images')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')

    args = parser.parse_args()

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
