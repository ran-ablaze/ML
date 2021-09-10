import argparse
from hw3_1 import Univariate_gaussian

parser = argparse.ArgumentParser()
parser.add_argument('--m', '--mean', type=float, default=0.0)
parser.add_argument('--s', '--var', type=float, default=1.0)
args = parser.parse_args()

def Sequential_Estimator():
    m = args.m
    s = args.s
    print('Data point source function: N({}, {})'.format(m, s))
    print()
    
    mean = 0
    var = 0
    num = 0

    while True:
        x = Univariate_gaussian(m, s)
        print('Add data point: {}'.format(x))
        num += 1
        p_mean = mean
        p_var = var
        mean += (x - p_mean) / num
        if num == 1:
            var = 0
        else:
            var += ((x - p_mean) * (x - mean) - var)/(num - 1)
        print('Mean = {:.15f}  Variance = {:.15f}'.format(mean, var))

        if abs(mean-p_mean) <= 0.01 and abs(var-p_var) <= 0.01:
            return

if __name__ == '__main__':
    Sequential_Estimator()