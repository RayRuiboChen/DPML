import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
import argparse
import datetime
import csv
import math
from datasets import *


def generate_train_data(train_tasks):
    print('Generating data...')
    input_x = []
    input_y = []

    for file_name in train_tasks.filelist:
        train_loader = train_tasks.tasks[file_name].trainloader
        x, y = train_loader

        input_x.append(x)
        input_y.append(y)

    input_x = torch.cat(input_x, dim=0)
    input_y = torch.cat(input_y, dim=0)
    print('Generating finished!')

    return input_x, input_y


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def print_running_info(args):
    print(f'current config: dataset:{args.dataset}, model type:{args.model_type}, '
          f'is baseline:{args.is_baseline}')

def read_filelist(filelist_name):
    train_list = []
    with open(filelist_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            train_list.append(line.strip())
    return train_list


def write_filelist(filelist_name, filelist):
    content = []
    for file in filelist:
        content.append(file + '\n')
    with open(filelist_name, 'w') as f:
        f.writelines(content)
    print(f"{len(content)} lines written.")
