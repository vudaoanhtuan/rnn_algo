import os
import argparse
import operator
import numpy as np
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('--num_sample', type=int, default=1000)
    parser.add_argument('--min', type=int, default=-999)
    parser.add_argument('--max', type=int, default=999)

    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = get_args()
    op = [operator.add, operator.sub, operator.mul]
    str_op = ['+', '-', '*']
    with open(args.output, 'w') as f:
        for i in tqdm(range(args.num_sample)):
            a = np.random.randint(args.min, args.max)
            b = np.random.randint(args.min, args.max)
            o = np.random.randint(0,3)
            c = op[o](a,b)
            s = str(a)+str(str_op[o])+str(b)+'=^'+str(c)+'$\n'
            f.write(s)
    
        