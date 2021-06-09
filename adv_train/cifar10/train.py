import argparse
import logging
import os
import sys
import time

import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preact_resnet import PreActResNet18
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgd, evaluate_standard)
from spade import *

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', default=45000, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=14, type=int)
    parser.add_argument('--small_epsilon', default=12, type=int)
    parser.add_argument('--eval_epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=7, type=int, help='Attack iterations')
    parser.add_argument('--iters', default=50, type=int, help='Eval Attack iterations')
    parser.add_argument('--restarts', default=10, type=int)
    parser.add_argument('--alpha', default=2, type=int, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='log', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--method', default='spade', type=str, choices=['spade', 'pgd', 'random'])
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()


def main():
    args = get_args()
    if args.method == "pgd":
        args.out_dir = args.method + "_{}".format(args.epsilon)
    else:
        args.out_dir = args.method + "_{}_{}".format(args.small_epsilon, args.epsilon)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    logfile = os.path.join(args.out_dir, f'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader, eps_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon_all = torch.cat(50000*[(args.epsilon/255. / std).view(1,3,1,1)])
    alpha = (args.alpha / 255.) / std

    model = PreActResNet18().cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # compute spade/random score
    # and assign epsilon accordingly
    if args.method != "pgd":
        ins = []
        outs = []
        ids = []
        for i, (img, _, ind) in enumerate(eps_loader):
            ins.append(img.numpy().reshape(-1,3072))
            outs.append(model(img.cuda()).detach().clone().cpu().numpy())
            ids.append(ind)
        ids = np.concatenate(ids, axis=0)
        sort_idx = np.argsort(np.asarray(ids))
        ins = np.concatenate(ins, axis=0)[sort_idx]
        outs = np.concatenate(outs, axis=0)[sort_idx]
        spade_score = spade(ins, outs)
        eps = args.epsilon*np.ones(50000)
        idx = (-spade_score).argsort()
        if args.method == "random":
            np.random.shuffle(idx)
        sub_idx = idx[args.topk:]
        eps[idx] = args.small_epsilon
        eps = torch.from_numpy(eps).float().cuda()
        epsilon_all = ((eps/255.).view(-1,1,1,1) / std.view(1,3,1,1))

    # Training
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y, idx) in enumerate(train_loader):
            X, y, idx = X.cuda(), y.cuda(), idx.cuda()
            epsilon = epsilon_all[idx]

            if args.method == "spade":
                clean_output = model(X)
                clean_index = torch.where(clean_output.max(1)[1]!=y)
                epsilon[clean_index] = 0

            delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                delta = 2*epsilon * torch.rand((X.size())).cuda() - epsilon
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            for _ in range(args.attack_iters):
                output = model(X + delta)
                loss = criterion(output, y)
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta.grad.zero_()
            delta = delta.detach()
            output = model(X + delta)
            loss = criterion(output, y)
            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)

    train_time = time.time()
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, args.iters, args.restarts, args.eval_epsilon)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


if __name__ == "__main__":
    main()
