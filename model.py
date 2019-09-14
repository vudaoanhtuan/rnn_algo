import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.emb(x)
        x = torch.relu(x)
        x, hs = self.rnn(x)
        return x, hs

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hs):
        x = self.emb(x)
        x = torch.relu(x)
        x, hs = self.rnn(x, hs)
        return x, hs

class Model(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, embed_size, hidden_size):
        super(Model, self).__init__()
        self.encoder = Encoder(enc_vocab, embed_size, hidden_size)
        self.decoder = Decoder(dec_vocab, embed_size, hidden_size)

    def encode(self, enc_inp):
        x, hs = self.encoder(enc_inp)
        return x, hs

    def decode(self, dec_inp, hs):
        x, hs = self.decoder(dec_inp, hs)
        return x, hs

    def forward(self, enc_inp, dec_inp):
        x, hs = self.encoder(enc_inp)
        x, hs = self.decoder(dec_inp, hs)
        return x, hs
    
    def forwar_and_loss(self, inp, lbl, pad_token=0):
        # inp: (batch_size, seq_len)
        # trg: (batch_size, seq_len)
        enc_inp = inp
        dec_inp = lbl[:, :-1]
        dec_trg = lbl[:, 1:]
        x, hs = self.encode(enc_inp)
        x, hs = self.decode(dec_inp, hs) # (batch_size, seq_len, vocab_size)
        x = x.reshape(-1, x.shape[-1])
        dec_trg = dec_trg.reshape(-1)
        loss = F.cross_entropy(x, dec_trg, ignore_index=pad_token)
        return loss

