import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
import argparse
import datetime
import meta_model
import csv
import math
from datasets import *
import utils
import baseline_models


def pre_train(model, epoch, batch_size, input_x, input_y, optimizer, is_linear=False):
    print('training...')
    model.train()
    train_loss = 0
    cnt = 0
    l = input_y.shape[0]
    order = list(range(l))
    random.shuffle(order)

    input_batch = []
    for i in range(l // batch_size):
        input_batch.append(
            (input_x[order[i * batch_size:(i + 1) * batch_size]], input_y[order[i * batch_size:(i + 1) * batch_size]])
        )

    for x1, y1 in input_batch:
        x1 = x1.cuda()
        y1 = y1.cuda()
        cnt += 1
        if is_linear:
            pred = model(x1)
        else:
            pred = model.feature_network(x1)
        y1 = y1.reshape((-1))

        loss = torch.mean((pred - y1) ** 2)
        train_loss += loss
        model.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= cnt
    print('Train Epoch: {} \tMSE: {:.6f}'.format(epoch, train_loss), file=open(args.log, 'a'), flush=True)
    print('Train Epoch: {} \tMSE: {:.6f}'.format(epoch, train_loss))
    return train_loss


def pre_test(model, device, test_tasks, test_list, is_linear=False):
    print('start testing')
    global args
    MSE, MAE, correct, cnt = 0, 0, 0, 0
    for file_name in test_list:
        model.eval()
        test_loader = test_tasks.tasks[file_name].testloader
        random.shuffle(test_loader)
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            if is_linear:
                output = model(x)
            else:
                output = model.feature_network(x)
            y = y.reshape((-1))
            b_size = x.shape[0]
            MSE += torch.sum(((output - y) ** 2), dim=0).item()
            MAE += torch.sum(((output - y).abs()), dim=0).item()
            correct += ((output - x[:, 59]) * (y - x[:, 59])).ge(0).float().sum().item()
            cnt += b_size
    MSE /= cnt
    MAE /= cnt
    correct /= cnt
    RMSE = math.sqrt(MSE)
    print('Test set: MSE: {:.6f}, RMSE: {:.6f}, MAE: {:.6f}, ACC: {:.6f} '.format(MSE, RMSE, MAE, correct),
          file=open(args.log, 'a'), flush=True)
    print('Test set: MSE: {:.6f}, RMSE: {:.6f}, MAE: {:.6f}, ACC: {:.6f} '.format(MSE, RMSE, MAE, correct), flush=True)
    return MSE, RMSE, MAE, correct


def pre_valid(model, device, test_tasks, test_list, is_linear=False):
    print('start validation')
    global args
    MSE, cnt = 0, 0
    for file_name in test_list:
        model.eval()
        test_loader = test_tasks.tasks[file_name].devloader
        random.shuffle(test_loader)
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            if is_linear:
                output = model(x)
            else:
                output = model.feature_network(x)
            b_size = x.shape[0]
            y = y.reshape((-1))
            MSE += torch.sum(((output - y) ** 2), dim=0).item()
            cnt += b_size

    MSE /= cnt
    print('Valid set: MSE: {:.6f}'.format(MSE), file=open(args.log, 'a'), flush=True)
    print('Valid set: MSE: {:.6f}'.format(MSE), flush=True)
    return MSE


def train(model, device, train_tasks, epoch, args):
    print('training...')
    model.train()
    train_loss = 0
    cnt = 0
    random.shuffle(train_tasks.filelist)
    for file_name in train_tasks.filelist:
        cnt += 1
        train_loader = train_tasks.tasks[file_name].trainloader
        x, y = train_loader
        x = x.to(device)
        y = y.to(device)
        train_loss += model.train_meta(x=x, y=y, code=file_name, device=device)

    train_loss /= cnt
    print('Train Epoch: {} \tMSE: {:.6f}'.format(epoch, train_loss), file=open(args.log, 'a'), flush=True)
    print('Train Epoch: {} \tMSE: {:.6f}'.format(epoch, train_loss))
    return train_loss


def test(model, device, test_tasks, test_list):
    print('start testing')
    global args

    MSE, MAE, correct, cnt = 0, 0, 0, 0
    for file_name in test_list:

        model.eval()
        test_loader = test_tasks.tasks[file_name].testloader
        random.shuffle(test_loader)
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            b_size = x.shape[0]
            output = model.evaluate(x, file_name)
            y = y.reshape((-1))
            MSE += torch.sum(((output - y) ** 2), dim=0).item()
            MAE += torch.sum(((output - y).abs()), dim=0).item()
            correct += ((output - x[:, 59]) * (y - x[:, 59])).ge(0).float().sum().item()
            cnt += b_size

    MSE /= cnt
    MAE /= cnt
    correct /= cnt
    RMSE = math.sqrt(MSE)
    print('Test set: MSE: {:.6f}, RMSE: {:.6f}, MAE: {:.6f}, ACC: {:.6f} '.format(MSE, RMSE, MAE, correct),
          file=open(args.log, 'a'), flush=True)
    print('Test set: MSE: {:.6f}, RMSE: {:.6f}, MAE: {:.6f}, ACC: {:.6f} '.format(MSE, RMSE, MAE, correct), flush=True)
    return MSE, RMSE, MAE, correct


def valid(model, device, test_tasks, test_list, eval_finetune=False):
    print('start validation')
    global args

    MSE, cnt = 0, 0
    for file_name in test_list:
        if eval_finetune:
            train_loader = test_tasks.tasks[file_name].trainloader
            x, y = train_loader
            x = x.to(device)
            y = y.to(device)
            model.eval_finetune(code=file_name, x=x, y=y)

        model.eval()
        test_loader = test_tasks.tasks[file_name].devloader
        random.shuffle(test_loader)
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            b_size = x.shape[0]
            output = model.evaluate(x, file_name)
            y = y.reshape((-1))
            MSE += torch.sum(((output - y) ** 2), dim=0).item()
            cnt += b_size

    MSE /= cnt
    print('Valid set: MSE: {:.6f}'.format(MSE), file=open(args.log, 'a'), flush=True)
    print('Valid set: MSE: {:.6f}'.format(MSE), flush=True)
    return MSE


def main():
    global args
    parser = argparse.ArgumentParser()

    # basics
    parser.add_argument('--log', default='', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--gpu_id', default='1', type=str)

    # meta learning
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--pretrain_max_epoch', default=50, type=int)
    parser.add_argument('--alpha', default=1e-4, type=float)
    parser.add_argument('--beta', default=1e-4, type=float)
    parser.add_argument('--gamma', default=1, type=float)
    parser.add_argument('--outer_lr', default=1e-5, type=float)
    parser.add_argument('--finetune_lr', default=1e-6, type=float)

    # parser.add_argument('--lr', default=1e-3, type=float)
    # parser.add_argument('--lr_decay', default=1.0, type=float)

    # model
    parser.add_argument('--dataset', default='five_minute', type=str)  # 'five_minute' or 'ten_minute'
    parser.add_argument('--model_type', default='linear', type=str)  # 'linear', 'lstm' or 'transformer'
    parser.add_argument('--input_size', default=200, type=int)
    parser.add_argument('--hidden_size', default=200, type=int)
    parser.add_argument('--num_layer', default=1, type=int) #number of LSTM layers
    parser.add_argument('--feature_size', default=30, type=int)
    parser.add_argument('--is_baseline', default=False, type=utils.str2bool)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    utils.set_seed(args.seed)

    utils.print_running_info(args)

    # set directories
    train_list_dic = {'five_minute': './filelist/five_minute_stock_list.txt',
                      'ten_minute': './filelist/ten_minute_stock_list.txt'}
    cache_dir_dic = {'five_minute': './data/five_minute_cache',
                     'ten_minute': './data/ten_minute_cache'}

    res_save_path = './results/' + args.dataset + '_' + args.model_type + '_meta_result.csv'

    if args.log == '':
        args.log = datetime.datetime.now().strftime("log/%Y-%m-%d-%H:%M:%S.txt")
    print(args, file=open(args.log, 'a'), flush=True)

    # read data
    train_list_filename = train_list_dic[args.dataset]
    train_list = utils.read_filelist(train_list_filename)

    train_tasks = Tasks(data_name=args.dataset, filelist=train_list,
                        cache_dir=cache_dir_dic[args.dataset], batch_size=32)
    train_tasks.checkdata()

    test_list = train_tasks.filelist
    device = torch.device("cuda")

    # init model
    if args.model_type == 'linear':
        if args.is_baseline:
            model = baseline_models.mlp()
        else:
            model = meta_model.DPML(model_type=args.model_type, emb_dim=160, hid_dim=300,
                                    alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                                    outer_lr=args.outer_lr, finetune_lr=args.finetune_lr)
    elif args.model_type == 'lstm':
        model = meta_model.DPML(model_type=args.model_type, emb_dim=30, hid_dim=80,
                                alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                                outer_lr=args.outer_lr, finetune_lr=args.finetune_lr)
    elif args.model_type == 'transformer':
        model = meta_model.DPML(model_type=args.model_type, emb_dim=200, hid_dim=400,
                                alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                                outer_lr=args.outer_lr, finetune_lr=args.finetune_lr)
        # specify emb_dim,hid_dim,param_dim required
    else:
        raise Exception('Invalid model type!')
    model = model.to(device)

    # pre training/baseline
    if args.model_type != 'linear' or args.is_baseline:
        if args.model_type == 'linear':
            is_linear = True
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
        else:
            is_linear = False
            optimizer = torch.optim.Adam(params=model.feature_network.parameters(), lr=1e-4)
        baseline_res_save_path = './results/' + args.dataset + '_' + args.model_type+'_baseline_result.csv'

        max_valid_MSE, max_MSE, max_RMSE, max_MAE, max_ACC = 1e10, 0, 0, 0, 0

        input_x, input_y = utils.generate_train_data(train_tasks)

        # scheduler = optim.lr_scheduler.ExponentialLR(meta_optimizer, args.lr_decay)
        for epoch in range(1, args.pretrain_max_epoch + 1):
            print(f'running epoch {epoch}')
            pre_train(model, epoch=epoch, batch_size=32, input_x=input_x, input_y=input_y,
                      optimizer=optimizer, is_linear=is_linear)
            # scheduler.step()
            valid_MSE = pre_valid(model, device, test_tasks=train_tasks, test_list=test_list, is_linear=is_linear)
            MSE, RMSE, MAE, ACC = pre_test(model, device, test_tasks=train_tasks,
                                           test_list=test_list, is_linear=is_linear)
            if valid_MSE < max_valid_MSE:
                max_valid_MSE, max_MSE, max_RMSE, max_MAE, max_ACC = valid_MSE, MSE, RMSE, MAE, ACC
                model.cpu()
                torch.save(model, args.log.replace('.txt', '.pt'))
                model.cuda()

        f = open(baseline_res_save_path, 'a', encoding='utf-8')
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            [args.dataset, args.model_type, args.seed, max_MSE, max_RMSE, max_MAE, max_ACC, max_valid_MSE])
        f.close()

        if args.is_baseline:
            return

        model = torch.load(args.log.replace('.txt', '.pt'))
        model = model.to(device)

    # our approach
    max_valid_MSE, max_MSE, max_RMSE, max_MAE, max_ACC = 1e10, 0, 0, 0, 0

    # scheduler = optim.lr_scheduler.ExponentialLR(meta_optimizer, args.lr_decay)

    for epoch in range(1, args.max_epoch + 1):
        print(f'running epoch {epoch}')
        train(model, device, epoch=epoch, train_tasks=train_tasks, args=args)
        # scheduler.step()

        valid_MSE = valid(model, device, test_tasks=train_tasks, eval_finetune=True, test_list=test_list)
        MSE, RMSE, MAE, ACC = test(model, device, test_tasks=train_tasks, test_list=test_list)

        if valid_MSE < max_valid_MSE:
            max_valid_MSE, max_MSE, max_RMSE, max_MAE, max_ACC = valid_MSE, MSE, RMSE, MAE, ACC

    f = open(res_save_path, 'a', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        [args.dataset, args.model_type, args.seed, max_MSE, max_RMSE, max_MAE, max_ACC, max_valid_MSE])
    f.close()


# torch.autograd.set_detect_anomaly(True)

main()
