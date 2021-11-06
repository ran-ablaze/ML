import numpy as np
import argparse
import matplotlib.pyplot as plt
import math
from scipy.special import expit
from scipy.linalg import inv

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=50)
parser.add_argument('--m', '--mean', nargs=4, type=float, default=0.0)
parser.add_argument('--v', '--var', nargs=4, type=float, default=1.0)
args = parser.parse_args()

def Univariate_gaussian(m, s):
    x = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
    #x~N(0,1)->#ax+b~N(b, a^2)
    return s**0.5 * x + m

def Generate_data(mx, vx, my, vy, n):
    data = np.zeros((n, 2))
    for i in range(n):
        data[i, 0] = Univariate_gaussian(mx, vx)
        data[i, 1] = Univariate_gaussian(my, vy)
    return data

def gradient_descent_regression(X, Y):
    w = np.random.rand(3, 1)
    count = 0
    while True:
        count += 1
        #wn+1 = wn + XT(yi-1/(1+e^Xw))
        deltaJ = X.T.dot(Y-expit(X.dot(w)))
        if (abs(deltaJ) < 1e-5).all() or count > 10000:
            return w
        w = w + deltaJ

def newton_regression(X, Y, n):
    w = np.random.rand(3,1)
    D = np.zeros((2*n, 2*n))
    count = 0
    while True:
        count += 1
        prev_w = w
        p = expit(X.dot(w))
        #Dii=e^-xw/(1+e^-xw)^2
        for i in range(2*n):
            D[i,i] = p[i].dot(1-p[i])
        #HF = XTDX
        Hessian_f = X.T.dot(D.dot(X))
        deltaf = X.T.dot(Y-p)
        # if np.linalg.det(Hessian_f)==0:
        #     w = w + deltaf
        # else:
        try:
            temp = w + inv(Hessian_f).dot(deltaf)
        except:
            temp = w +deltaf
            # w = np.random.rand(3,1)
        w = temp
        if (abs(w-prev_w) < 1e-5).all() or count > 10000:
            return w

def classfication(X, w):
    print('w:\n{:15.10f}\n{:15.10f}\n{:15.10f}\n'.format(w[0,0], w[1,0], w[2,0]))
    print('Confusion Matrix:')
    class1=[]
    class2=[]
    tp = 0
    tn = 0
    for i in range(len(X)):
        if X[i].dot(w) < 0:
            class1.append(X[i, 0:2])
            if i < n:
                tp+=1
        else:
            class2.append(X[i, 0:2])
            if i >= n:
                tn+=1
    class1 = np.array(class1)
    class2 = np.array(class2)
    print('\t    |Predict cluster 1|Predict cluster 2')
    print('Is cluster 1|\t   ',tp,'\t      |\t     ',n-tp)
    print('Is cluster 2|\t   ',n-tn,'\t      |\t     ',tn)
    print()
    print('Sensitivity (Successfully predict cluster 1): {:7.5f}'.format(tp/n))
    print('Specificity (Successfully predict cluster 2): {:7.5f}'.format(tn/n))
    return class1, class2

def Visualization(X, w1, w2, n):
    plt.subplot(131)
    plt.title("Ground truth")
    plt.scatter(X[0:n, 0], X[0:n, 1], c='r')
    plt.scatter(X[n:2*n, 0], X[n:2*n, 1], c='b')

    print('Gradient descent:\n')
    class1, class2 = classfication(X, w1)
    plt.subplot(132)
    plt.title("Gradient descent")
    plt.scatter(class1[:,0], class1[:,1], c='r')
    plt.scatter(class2[:,0], class2[:,1], c='b')

    print('\n----------------------------------------')
    print("Newton's method:\n")
    # print(w2)
    class1, class2 = classfication(X, w2)
    plt.subplot(133)
    plt.title("Newton's method:")
    plt.scatter(class1[:,0], class1[:,1], c='r')
    plt.scatter(class2[:,0], class2[:,1], c='b')
    plt.show()

if __name__ == '__main__':
    n = args.n
    mx1 = args.m[0]
    my1 = args.m[1]
    mx2 = args.m[2]
    my2 = args.m[3]
    vx1 = args.v[0]
    vy1 = args.v[1]
    vx2 = args.v[2]
    vy2 = args.v[3]

    D1 = Generate_data(mx1, vx1, my1, vy1, n)
    D2 = Generate_data(mx2, vx2, my2, vy2, n)

    #Xw = w1x + w2y +w3, X=[x, y, 1]
    #Y = Bernoulli(f(Xw))
    X = np.zeros((2 * n, 3))
    Y = np.zeros((2 * n, 1), dtype=int)
    X[0:n, 0:2] = D1
    X[n:2*n, 0:2] = D2
    X[:, 2] = 1
    Y[n:2*n, 0] = 1

    w1 = gradient_descent_regression(X, Y)
    w2 = newton_regression(X, Y, n)
    Visualization(X, w1, w2, n)
