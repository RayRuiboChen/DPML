import torch
import torch.nn as nn
import numpy as np
import random
import os
import pandas as pd
import copy
import torch.nn.functional as F
import math


class mlp(nn.Module):
    def __init__(self, emb_dim=160):
        super().__init__()
        self.fc = nn.Linear(in_features=emb_dim, out_features=1)
        pass

    def forward(self, x):
        '''
        x: [b,emb_dim]
        '''
        pred = self.fc(x)
        pred = pred.reshape((-1))
        return pred


class Attentive_Pooling(nn.Module):
    def __init__(self, hidden_size):
        super(Attentive_Pooling, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size)
        self.u = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, memory, mask=None):
        h = torch.tanh(self.w_1(memory))
        score = torch.squeeze(self.u(h), -1)  # node,
        if mask is not None:
            score = score.masked_fill(mask.eq(0), -1e9)
        alpha = F.softmax(score, -1)  # node,
        s = torch.sum(torch.unsqueeze(alpha, -1) * memory, -2)
        return s


class LSTM(nn.Module):
    def __init__(self, input_size=200, hidden_size=200, num_layer=1, feature_size=30, full_chlov=True,
                 attn_pooling=True):
        super(LSTM, self).__init__()
        self.full_chlov = full_chlov
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.feature_size = feature_size
        self.attn_pooling = attn_pooling

        if self.full_chlov:
            self.linear_chlov = nn.Linear(5, input_size)
            self.linear_history = nn.Linear(5, input_size)
        else:
            self.linear_chlov = nn.Linear(1, input_size)
            self.linear_history = nn.Linear(1, input_size)

        self.lstm_chlov = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)
        self.lstm_history = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)

        if self.attn_pooling:
            self.attn_pooling_chlov = Attentive_Pooling(hidden_size)
            self.attn_pooling_history = Attentive_Pooling(hidden_size)

        if feature_size != 0:
            self.linear_out1 = nn.Linear(2 * hidden_size, feature_size)
            self.linear_out2 = nn.Linear(feature_size, 1)
        else:
            self.linear_out = nn.Linear(2 * hidden_size, 1)

    def forward(self, x, early_exit=False):
        chlov = x[:, :60].reshape((-1, 12, 5))
        history = x[:, 60:].reshape((-1, 20, 5))
        self.lstm_chlov.flatten_parameters()
        self.lstm_history.flatten_parameters()
        if self.full_chlov:
            chlov = F.relu(self.linear_chlov(chlov))
            history = F.relu(self.linear_history(history))
        else:
            chlov = F.relu(self.linear_chlov(chlov[:, :, -1:]))
            history = F.relu(self.linear_history(history[:, :, -1:]))

        B, L, H = chlov.size()
        chlov, _ = self.lstm_chlov(chlov)
        history, _ = self.lstm_history(history)

        if self.attn_pooling:
            chlov = self.attn_pooling_chlov(chlov)
            history = self.attn_pooling_history(history)
        else:
            chlov = chlov[:, -1, :]
            history = history[:, -1, :]

        x = torch.cat((chlov, history), dim=1)

        if self.feature_size != 0:
            x = F.relu(self.linear_out1(x))
            if early_exit:
                return x
            x = self.linear_out2(x)
        else:
            x = self.linear_out(x)

        x = x.reshape((-1))
        return x


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Transformer(nn.Module):
    def __init__(self, input_size=200, hidden_size=200, num_layer=6):
        super(Transformer, self).__init__()
        c = copy.deepcopy
        d_model = hidden_size
        head_num = 8
        dropout = 0.1
        self.dropout = nn.Dropout(dropout)
        attn = MultiHeadedAttention(head_num, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_model, dropout)
        self.position_emb = nn.Embedding(20 + 12 + 1, input_size)
        self.word_lienar = nn.Linear(5, input_size)
        self.emb_proj = nn.Linear(input_size, d_model)
        self.encoder = trm_Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), num_layer)
        self.w_out = nn.Linear(d_model, 1)
        self.bos_emb = nn.Parameter(torch.zeros(1, 1, 5))
        self.init()

    def init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, early_exit=False):
        chlov = x[:, :60].reshape(-1, 12, 5)
        history = x[:, 60:].reshape(-1, 20, 5)

        batch_size, seq_len = chlov.size(0), 20 + 12 + 1
        device = chlov.device

        input = torch.cat((self.bos_emb.repeat(batch_size, 1, 1), chlov, history), dim=1)
        bos = torch.ones(input.size(0), 1, device=device).long().fill_(2)
        word_mask = torch.ones(batch_size, seq_len, device=device).long()
        pos_indices = torch.unsqueeze(torch.arange(seq_len), 0).repeat(batch_size, 1).to(device)
        word_emb = self.word_lienar(input)
        pos_emb = self.position_emb(pos_indices)
        emb = self.emb_proj(word_emb + pos_emb)

        hidden = self.encoder(emb, word_mask)

        if early_exit:
            return hidden[:, 0, :]

        result = self.w_out(self.dropout(hidden[:, 0, :])).reshape((-1))

        return result


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class trm_Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(trm_Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        if mask.dim() == 2:
            seq_len = mask.size(1)
            mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            assert mask.size() == (x.size(0), seq_len, seq_len)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity we apply the norm first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer function that maintains the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of two sublayers, self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=0.0):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.eq(0), -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # (Dropout described below)
    p_attn = F.dropout(p_attn, p=dropout)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # clone linear for 4 times, query, key, value, output
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            assert mask.dim() == 4  # batch, head, seq_len, seq_len
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => head * d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.p)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, emb_size, vocab, emb=None):
        super(Embeddings, self).__init__()
        if emb is not None:
            self.lut = emb
        else:
            self.lut = nn.Embedding(vocab, emb_size)
        self.emb_size = emb_size
        self.criterion = nn.CosineEmbeddingLoss(margin=0.5)

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    # TODO: use learnable position encoding
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
