import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import torch
from torch.utils.data import TensorDataset, DataLoader


def get_data_loader(tokenizer, data_file, batch_size=32):
    with open(data_file) as f:
        data = f.read().split('\n')[:-1]

    inp = []
    trg = []
    for s in data:
        x,y = s.split('=')
        inp.append(x)
        trg.append(y)
    
    inp = tokenizer.texts_to_sequences(inp)
    trg = tokenizer.texts_to_sequences(trg)
    inp = pad_sequences(inp, padding='post')
    trg = pad_sequences(trg, padding='post')

    inp = torch.tensor(inp, dtype=torch.long)
    trg = torch.tensor(trg, dtype=torch.long)

    ds = TensorDataset(inp, trg)
    dl = DataLoader(ds, batch_size=batch_size)
    return dl
