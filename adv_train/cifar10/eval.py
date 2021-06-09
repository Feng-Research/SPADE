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
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--eval_epsilon', default=8, type=int)
    parser.add_argument('--epsilon', default=14, type=int)
    parser.add_argument('--small_epsilon', default=12, type=int)
    parser.add_argument('--iters', default=50, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=10, type=int)
    parser.add_argument('--method', default='spade', type=str, choices=['spade', 'pgd', 'random'])
    parser.add_argument('--out-dir', default='log', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
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

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    _, test_loader, _ = get_loaders(args.data_dir, args.batch_size)

    # Evaluation
    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(torch.load(os.path.join(args.out_dir, 'model.pth')))
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, args.iters, args.restarts, args.eval_epsilon)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)

if __name__ == "__main__":
    main()
