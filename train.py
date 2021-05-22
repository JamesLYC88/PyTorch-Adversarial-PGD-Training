from data import GetTrainLoader, GetTestLoader
from utils import fix_seeds, clear_line, pgd
from pytorchcv.model_provider import get_model as ptcv_get_model

import torch
import torch.nn as nn

import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def epoch_adversarial(args, model, loader, criterion, optimizer=None):
    if optimizer:
        model.train()
        state = 'training...'
    else:
        model.eval()
        state = 'testing...'
    train_acc, train_loss = 0.0, 0.0
    for i, (x, y) in enumerate(loader):
        print(f'{state} {i+1}/{len(loader)}', end='\r')
        x, y = x.to(device), y.to(device)
        delta = pgd(model, x, y, criterion, args.eps/255/std, args.alpha/255/std, args.num_iter)
        yp = model(x+delta)
        loss = criterion(yp, y)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
    clear_line()
    return train_acc / len(loader.dataset), train_loss / len(loader.dataset)

def main(args):

    # pre
    os.mkdir(args.save_dir)
    fix_seeds()

    # data
    train_loader = GetTrainLoader(args.bs)
    test_loader = GetTestLoader(args.bs)

    # model / criterion / optimizer
    model = ptcv_get_model(args.model, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # train / test
    best_acc = 0.0
    for e in range(args.epoch):
        start = time.time()
        # train
        train_acc, train_loss = epoch_adversarial(args, model, train_loader, criterion, optimizer)
        # test
        test_acc, test_loss = epoch_adversarial(args, model, test_loader, criterion)
        # log
        m, s = divmod(int(time.time()-start), 60)
        log = f'[{e+1:03d}/{args.epoch:03d}] ({m} min {s} sec)'
        log += f' | train_acc = {train_acc:.5f} train_loss = {train_loss:.5f}'
        log += f' | test_acc = {test_acc:.5f} test_loss = {test_loss:.5f}'
        log += ' *' if test_acc > best_acc else ''
        print(log)
        # save
        suffix = f'{int(test_acc*1e3)}.pth'
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_{suffix}'))
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch_{e+1}_{suffix}'))

if __name__ == '__main__':

    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bs', type=int, default=128, help='data: batch size')
    parser.add_argument('--lr', type=float, default=1e-1, help='train: learning rate')
    parser.add_argument('--model', type=str, default='wrn16_10_cifar10', help='model')
    parser.add_argument('--epoch', type=int, default=250, help='train: epoch')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--save_dir', type=str, default='ckpt', help='checkpoint')

    parser.add_argument('--eps', type=float, default=8, help='pgd: epsilon')
    parser.add_argument('--alpha', type=float, default=0.8, help='pgd: alpha')
    parser.add_argument('--num_iter', type=int, default=20, help='pgd: num_iter')

    args = parser.parse_args()

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # global
    cifar_10_mean = (0.491, 0.482, 0.447)
    cifar_10_std = (0.202, 0.199, 0.201)
    mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
    std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

    main(args)
