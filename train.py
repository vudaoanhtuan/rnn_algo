import os
import argparse
import pickle

import torch
import torch.nn as nn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from model import Model
from data_loader import get_data_loader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('valid_file')
    parser.add_argument('token_file')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--load_weight')
    parser.add_argument('--weight_dir', default='weight')

    args = parser.parse_args()
    return args

def train(model, optim, dl, device=torch.device('cpu')):
    model.train()
    with tqdm(total=len(dl)) as pbar:
        for inp, trg in dl:
            inp = inp.to(device)
            trg = trg.to(device)
            optim.zero_grad()
            loss = model.forwar_and_loss(inp, trg)
            loss.backward()
            optim.step()
            
            pbar.update(1)
            pbar.set_description('loss     = %.6f' % loss.item())

def eval(model, dl, device=torch.device('cpu')):
    model.eval()
    with tqdm(total=len(dl)) as pbar, torch.no_grad():
        for inp, trg in dl:
            inp = inp.to(device)
            trg = trg.to(device)
            loss = model.forwar_and_loss(inp, trg)
            
            pbar.update(1)
            pbar.set_description('val_loss = %.6f' % loss.item())


if __name__=='__main__':
    args = get_args()

    with open(args.token_file, 'rb') as f:
        tokenizer = pickle.load(f)

    train_dl = get_data_loader(tokenizer, args.train_file, args.batch_size)
    val_dl = get_data_loader(tokenizer, args.valid_file, args.batch_size)

    vocab_size = len(tokenizer.word_index) + 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(vocab_size, vocab_size, args.emb_size, args.hidden_size)
    model = model.to(device)

    if args.load_weight is not None:
        print("Load weight from %s" % args.load_weight)
        w = torch.load(args.load_weight)
        model.load_state_dict(w)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)


    if not os.path.isdir(args.weight_dir):
        os.mkdir(args.weight_dir)

    for e in range(args.num_epoch):
        train(model, optim, train_dl, device)
        eval(model, val_dl, device)
        torch.save(model.state_dict(), os.path.join(args.weight_dir, 'model_e%02d.h5'%e))

    
