import collections
import hashlib
import math
import os
import random
import tarfile
import time
import zipfile
import paddle.device.cuda
import requests
import numpy as np
from paddle import nn
from Vocab import Vocab

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

DATA_HUB['ptb'] = (DATA_URL + 'ptb.zip',
                   '319d85e578af0cdc590547f26231e4e31cdf1e42')


def download(name, cache_dir=os.path.join('./', 'data')):
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    for name in DATA_HUB:
        download(name)


def try_gpu(i=0):
    if paddle.device.cuda.device_count() >= i + 1:
        return paddle.CUDAPlace(i)
    return paddle.CPUPlace()


def try_all_gpus():
    devices = [paddle.CUDAPlace(i)
               for i in range(paddle.device.cuda.device_count())]
    return devices if devices else paddle.CPUPlace()


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


def read_ptb():
    data_dir = download_extract('ptb')
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]


def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def subsample(sentences, vocab):
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())

    def keep(token):
        return (random.uniform(0, 1) <
                math.sqrt(1e-4 / counter[token] * num_tokens))

    return [[token for token in line if keep(token)] for line in sentences], counter


def get_centers_and_contexts(corpus, max_window_size):
    centers, contexts = [], []
    for line in corpus:
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts


class RandomGenerator:
    def __init__(self, sampling_weights):
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            self.candidates = random.choices(self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


def get_negatives(all_contexts, vocab, counter, K):
    sampling_weights = [counter[vocab.to_tokens(i)] ** 0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += \
            [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (paddle.to_tensor(centers).reshape((-1, 1)), paddle.to_tensor(
        contexts_negatives), paddle.to_tensor(masks), paddle.to_tensor(labels))


def load_data_ptb(batch_size, max_window_size, num_noise_words):
    sentences = read_ptb()
    vocab = Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
    all_negatives = get_negatives(all_contexts, vocab, counter, 5)

    class PTBDataset(paddle.io.Dataset):
        def __init__(self, centers, contexts, negatives):
            # super().__init__()
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return self.centers[index], self.contexts[index], self.negatives[index]

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = paddle.io.DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True, collate_fn=batchify)
    return data_iter, vocab


def skip_gram(center, contexts_and_negatices, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatices)
    pred = paddle.bmm(v, u.transpose(perm=[0, 2, 1]))
    return pred


class SigmoidBCELoss(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            logit=inputs, label=target, weight=mask, reduction="none")
        return out.mean(axis=1)


def train(net, data_iter, lr, loss, num_epochs, device=try_gpu(0)):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.initializer.XavierUniform(m.weight)

    net.apply(init_weights)
    optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=net.parameters())
    metric = Accumulator(2)
    for epoch in range(num_epochs):
        timer = Timer()
        for i, batch in enumerate(data_iter):
            optimizer.clear_grad()
            center, context_negative, mask, label = [
                paddle.to_tensor(data, place=device) for data in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape), paddle.to_tensor(label, dtype='float32'),
                      paddle.to_tensor(mask, dtype='float32'))
                 / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')


def get_similar_tokens(query_token, vocab, k, embed):
    W = embed.weight
    x = W[vocab[query_token]]
    cos = paddle.mv(W, x) / paddle.sqrt(paddle.sum(W * W, axis=1) *
                                        paddle.sum(x * x) + 1e-9)
    topk = paddle.topk(cos, k=k + 1)[1].numpy().astype('int32')
    for i in topk[1:]:
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')
