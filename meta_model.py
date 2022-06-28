import torch
import torch.nn as nn
import numpy as np
import random
import os
import pandas as pd
import copy
import baseline_models


class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim):
        super().__init__()
        mid_rate = (hid_dim / emb_dim) ** (1 / 3)
        self.fc1 = nn.Linear(in_features=emb_dim, out_features=round(emb_dim * mid_rate))
        self.fc2 = nn.Linear(in_features=round(emb_dim * mid_rate), out_features=round(emb_dim * mid_rate ** 2))
        self.fc3 = nn.Linear(in_features=round(emb_dim * mid_rate ** 2), out_features=round(hid_dim))
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        params:
        x: [b,emb_dim]
        y: [b]
        return:
        z: [1,hid_dim]
        '''
        z0 = self.relu(self.fc1(x))  # [b,hid_dim]
        z0 = self.relu(self.fc2(z0))
        z0 = self.fc3(z0)
        z = torch.mean(z0, dim=0, keepdim=True)
        return z


class Decoder(nn.Module):
    def __init__(self, hid_dim, param_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.param_dim = param_dim

        mid_rate = (param_dim / hid_dim) ** (1 / 3)
        self.fc1 = nn.Linear(in_features=hid_dim, out_features=round(hid_dim * mid_rate))
        self.fc2 = nn.Linear(in_features=round(hid_dim * mid_rate), out_features=round(hid_dim * mid_rate ** 2))
        self.fc3 = nn.Linear(in_features=round(hid_dim * mid_rate ** 2), out_features=param_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        '''
        param:
        z: [b,hid_dim] b maybe 1
        return:
        params: [b,param_dim]
        '''
        params = self.relu(self.fc1(z))
        params = self.relu(self.fc2(params))
        params = self.fc3(params)
        return params


class DPML(nn.Module):
    def __init__(self, emb_dim=160, hid_dim=300,
                 alpha=1e-4, beta=1e-4, gamma=1, outer_lr=1e-5, finetune_lr=1e-6,
                 inner_max_epoch=5, finetune_max_epoch=1, model_type='linear'):
        '''
        params for five_minute:
        hid_dim=300
        outer_lr=1e-5
        inner_lr=1e-3
        latent_lr=1e-3
        '''
        super().__init__()
        param_dim = emb_dim + 1

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.encoder = Encoder(emb_dim=emb_dim, hid_dim=hid_dim)
        self.decoder = Decoder(hid_dim=hid_dim, param_dim=param_dim)
        self.model_type = model_type

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.finetune_lr = finetune_lr
        self.inner_max_epoch = inner_max_epoch
        self.outer_lr = outer_lr

        self.finetune_max_epoch = finetune_max_epoch
        self.criterion = nn.MSELoss(reduction='mean')

        model_params = [x[1] for x in list(filter(
            lambda kv: kv[0].split('.')[0] != 'decoder', self.named_parameters()))]
        self.param_opt = torch.optim.Adam(params=model_params, lr=self.outer_lr, weight_decay=0)

        self.relu = nn.ReLU()
        self.latents = {}
        self.params = {}

        self.code_set = set()

        if model_type == 'linear':
            pass
        elif model_type == 'lstm':
            self.feature_network = baseline_models.LSTM()
        elif model_type == 'transformer':
            self.feature_network = baseline_models.Transformer()
        else:
            raise Exception('Invalid model type!')

    def extract_feature(self, x):
        if self.model_type != 'linear':
            return self.feature_network(x, early_exit=True).detach()
        else:
            return x

    def predict(self, x, params):
        '''
        x: [b,emb_dim]
        params:[1,param_dim]
        '''
        w = params[:, :self.emb_dim]
        b = params[:, self.emb_dim]
        pred = torch.sum(x * w, dim=1, keepdim=False) + b  # [b]

        return pred

    def sample_data(self, total_num, batch_size):
        if total_num <= batch_size:
            return list(range(total_num))
        st = random.randint(0, total_num - batch_size)
        return list(range(st, st + batch_size))

    def adjust_decoder(self, tgt):
        d = dict(self.decoder.named_parameters())
        d_tgt = dict(tgt.named_parameters())
        for name in d:
            d[name].data = d[name].data + (d_tgt[name].data - d[name].data) * self.gamma
            d[name].data = d[name].data.clone().detach()

    def train_meta(self, x, y, code, device=None, batch_size=32, train_step=5):
        total_num = x.shape[0]
        total_loss = 0
        decoder = copy.deepcopy(self.decoder)
        opt = torch.optim.SGD(params=decoder.parameters(), lr=1e-5)

        for i in range(train_step):
            train_list = self.sample_data(total_num, batch_size)
            dev_list = self.sample_data(total_num, batch_size)
            x0 = self.extract_feature(x[train_list])
            y0 = y[train_list]

            self.inner_layer(x0, y0, code, decoder=decoder)
            x1 = self.extract_feature(x[dev_list])
            y1 = y[dev_list]

            decoder.zero_grad()
            total_loss += self.outer_layer(x1, y1, code, device=device, opt=opt)

        self.code_set.add(code)
        self.adjust_decoder(tgt=decoder)

        total_loss /= train_step
        return total_loss

    def inner_layer(self, x, y, code, decoder):
        '''
        params:
        x: [b,emb_dim] ,D^{Tr}
        y: [b]
        '''
        if code not in self.latents:
            self.latents[code] = torch.zeros((1, self.hid_dim)).cuda()

        z = self.encoder(x)
        for epoch in range(self.inner_max_epoch):
            z.retain_grad()
            params = decoder(z)
            pred = self.predict(x, params)  # [b]
            y = y.reshape((-1))
            loss = torch.mean(((pred - y) ** 2))
            self.zero_grad()
            loss.backward(retain_graph=True)
            z = z - z.grad.data * self.alpha

        emb = self.latents[code].data.clone().detach()
        grad = z - emb
        self.latents[code] = self.latents[code].data.clone().detach() + grad * self.beta

        params = decoder(self.latents[code])
        self.params[code] = params

    def outer_layer(self, x, y, code, device, opt):
        x = x.to(device)
        y = y.to(device)
        batch_size = x.shape[0]
        pred = self.predict(x, self.params[code])  # [b]

        y = y.reshape((-1))
        loss = torch.sum((pred - y) ** 2)

        loss = loss / batch_size
        self.zero_grad()
        loss.backward()

        self.param_opt.step()
        opt.step()

        return loss

    def eval_finetune(self, code, x, y, steps=10, batch_size=32):
        self.train()
        total_num = x.shape[0]

        decoder = copy.deepcopy(self.decoder)
        opt = torch.optim.Adam(params=decoder.parameters(), lr=self.finetune_lr)
        criterion = nn.MSELoss(reduction='mean')
        for step in range(steps):
            train_list = self.sample_data(total_num, batch_size)
            x1 = self.extract_feature(x[train_list])
            y1 = y[train_list]
            y1 = y1.reshape((-1))
            params = decoder(self.latents[code].detach())
            pred = self.predict(x1, params)
            loss = criterion(y1, pred)
            self.zero_grad()
            loss.backward()
            opt.step()

        params = decoder(self.latents[code])
        self.params[code] = params

    def evaluate(self, x, code):
        pred = self.predict(self.extract_feature(x), self.params[code])
        return pred
