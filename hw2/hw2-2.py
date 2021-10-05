import os
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--a", type=int, default=0)
parser.add_argument("--b", type=int, default=0)
parser.add_argument('--filename', type=str, default='testfile.txt')
args = parser.parse_args()

def count_outcome(outcomes):
    out_a = 0
    out_b = 0
    for i in outcomes:
        if i == '1':
            out_a += 1
        else:
            out_b += 1
    return out_a, out_b

def beta(a, b):
    p = a/(a+b)
    return p**a*(1-p)**b*math.factorial(a+b)/(math.factorial(a)*math.factorial(b)) 

if __name__ == '__main__':
    prior_a = args.a
    prior_b = args.b
    fp = open(args.filename, "r")
    i = 1
    outcomes = fp.readline().strip()
    while outcomes:
        cnt_a, cnt_b = count_outcome(outcomes)
        posterior_a = prior_a + cnt_a
        posterior_b = prior_b + cnt_b
        likelihood = beta(cnt_a, cnt_b)
        print('case', i, ':', outcomes)
        print('Likelihood:',likelihood)
        print('Beta prior:\ta =',prior_a,'b =', prior_b)
        print('Beta posterior: a =',posterior_a,'b =', posterior_b)
        prior_a = posterior_a
        prior_b = posterior_b
        outcomes = fp.readline().strip()
        i += 1
    fp.close()
