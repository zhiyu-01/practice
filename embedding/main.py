
from utils import *


data_iter, vocab = load_data_ptb(512, 5, 5)
loss = SigmoidBCELoss()

embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))

lr, num_epoch = 0.002, 5
train(net, data_iter, lr, loss, num_epoch)

get_similar_tokens('chip', vocab, 30, net[0])
