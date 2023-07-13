import os
import paddle
import paddle.nn as nn
import requests
import pandas as pd
import re
import warnings
import numpy as np

warnings.filterwarnings('ignore')


def _text_standardize(text):
    text = re.sub("\d+(,\d+)?(\.\d+)?", "<NUM>", text)
    text = re.sub("\d+-+?\d*", "<NUM>", text)
    return text.strip()


files = os.listdir('./MRPC')
data = {'train': None, 'test': None}
for f in files:
    path = os.path.join('./MRPC', f)
    df = pd.read_csv(path, sep='\t', error_bad_lines=False)
    label = df.iloc[:, 0].values
    s1 = df.iloc[:, 3].values
    s2 = df.iloc[:, 4].values
    if 'train' in f:
        data['train'] = {'label': label, 's1': s1, 's2': s2}
    else:
        data['test'] = {'label': label, 's1': s1, 's2': s2}

vocab = set()
for n in ['train', 'test']:
    for m in ['s1', 's2']:
        for i in range(len(data[n][m])):
            text = data[n][m][i]
            text = str(text)
            text = _text_standardize(text).split(' ')
            vocab.update(set(text))

v2i = {v: i for i, v in enumerate(vocab, start=1)}
v2i['<PAD>'] = 0
v2i['<UNK>'] = len(v2i)
v2i['<SEP>'] = len(v2i)
v2i['<GO>'] = len(v2i)
i2v = {i: v for v, i in v2i.items()}

for n in ['train', 'test']:
    for m in ['s1', 's2']:
        data[n][m + 'id'] = [[v2i.get(v, v2i['<UNK>']) for v in str(s).split(' ')] for s in data[n][m]]


class Dataset(paddle.io.Dataset):
    def __init__(self, data, i2v, v2i, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.i2v = i2v
        self.v2i = v2i
        sequence = data['train']['s1id'] + data['train']['s2id']
        sequence = [[v2i['<GO>']] + s + [v2i['<SEP>']] for s in sequence]
        len_seq = len(sequence)
        max_length = max([len(s) for s in sequence])
        self.full_data = np.full((len_seq, max_length), fill_value=pad_id, dtype='int32')

        for i, s in enumerate(sequence):
            l = len(s)
            self.full_data[i, :l] = s

    def __getitem__(self, idx):
        return self.full_data[idx]

    def __len__(self):
        return len(self.full_data)


train_data = Dataset(data, i2v, v2i)


class ELMo_Model(nn.Layer):
    def __init__(self, voc_size, output_size, emb_size=64, n_layers=2, lr=2e-3):
        super().__init__()
        self.voc_size = voc_size
        self.emb = nn.Embedding(voc_size, emb_size, padding_idx=0,
                                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(0, 0.1)))
        self.f_lstms = nn.LayerList([nn.LSTM(emb_size, output_size, time_major=False) if i == 0 else nn.LSTM(
            output_size, output_size, time_major=False) for i in range(n_layers)])
        self.b_lstms = nn.LayerList([nn.LSTM(emb_size, output_size, time_major=False) if i == 0 else nn.LSTM(
            output_size, output_size, time_major=False) for i in range(n_layers)])
        self.f_linear = nn.Linear(output_size, voc_size)
        self.b_linear = nn.Linear(output_size, voc_size)

    def forward(self, seq):
        emb = self.emb(seq)
        f_outs = [emb[:, :-1, :]]
        b_outs = [paddle.flip(emb[:, 1:, :], axis=1)]
        for f_lstm in self.f_lstms:
            f_out, (h_, c_) = f_lstm(f_outs[-1])
            f_outs.append(f_out)

        for b_lstm in self.b_lstms:
            b_out, (h_, c_) = b_lstm(b_outs[-1])
            b_outs.append(b_out)

        f_out_l = self.f_linear(f_outs[-1])
        b_out_l = self.b_linear(b_outs[-1])
        return f_out_l, b_out_l

    def get_emb(self, seq):
        fo, bo = self(seq)
        embs = [paddle.concat([f[:, 1:, :], paddle.flip(b, axis=1)[:, :-1, :]], axis=2) for f, b in zip(fo, bo)]
        for i, emb in enumerate(embs, start=1):
            print('第{}的词向量维度为{}'.format(i, emb.shape[2]))
        return embs


epoch = 100
voc_size = len(v2i)
output_size = 128
batch_size = 64
lr = 2e-3

dataloader = paddle.io.DataLoader(train_data, batch_size=batch_size, shuffle=True)
elmo = ELMo_Model(voc_size=voc_size, output_size=output_size)
opt = paddle.optimizer.Adam(learning_rate=lr, parameters=elmo.parameters())
loss = nn.CrossEntropyLoss()

for i in range(epoch):
    for batch, data in enumerate(dataloader()):
        f_outs, b_outs = elmo(data)

        fo = paddle.reshape(f_outs, (-1, voc_size))
        bo = paddle.reshape(b_outs, (-1, voc_size))

        f_label = paddle.reshape(data[:, 1:], (-1,))
        b_label = paddle.reshape(paddle.flip(data[:, :-1], axis=1), (-1))
        f_label = paddle.cast(f_label, dtype='int64')
        b_label = paddle.cast(b_label, dtype='int64')

        l = (loss(fo, f_label) + loss(bo, b_label)) / 2
        l.backward()
        opt.step()
        opt.clear_grad()

        fo = f_outs[0].zrgmax(axis=1).numpy()
        bo = paddle.flip(b_outs, axis=1)[0].argmax(axis=1).numpy()
    if (i + 1) % 10 == 0:
        print('\n\nEpoch:{}, batch:{}, loss:{:.4f}'.format(i + 1, batch + 1, loss.item()),
              '\ntarget:{}'.format(' '.join([i2v[j] for j in data[0].numpy() if j != train_data.pad_id])),
              '\nforward:{}'.format(' '.join([i2v[j] for j in fo if j != train_data.pad_id])),
              '\nbackward:{}'.format(' '.join([i2v[j] for j in bo if j != train_data.pad_id])),
              )
        paddle.save(elmo.state_dict(), './work/elmo_{}.pdparams'.format(i + 1))
