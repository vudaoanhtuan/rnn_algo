import argparse
import pickle

from keras.preprocessing.text import Tokenizer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--output', default='tokenizer.pkl')

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()
    tokenizer = Tokenizer(char_level=True)
    with open(args.file) as f:
        data = f.read().split('\n')[:-1]
        tokenizer.fit_on_texts(data)
    with open(args.output, 'wb') as f:
        pickle.dump(tokenizer, f)
        print("tokenizer saved at %s" % args.output)