import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
import time

def read_data(data_file='input.data'):
    with open(data_file, 'rb') as file:
        lines = file.readlines()
    file.close()

    datas = np.array([line.split()
                    for line in lines], dtype="float64")
    X = datas[:, 0].reshape(-1, 1)
    Y = datas[:, 1].reshape(-1, 1)
    return X, Y

def rational_quadratic_kernel(xi, xj, l, alpha):
    return (1+cdist(xi, xj, 'sqeuclidean')/(2*alpha*l**2))**(-alpha)

def negative_marginal_likelihood(theta):
    C = rational_quadratic_kernel(X, X, theta[0], theta[1]) + 1/beta*np.identity(X.shape[0])
    res = 0.5*np.log(np.linalg.det(C))+0.5*Y.T.dot(np.linalg.inv(C).dot(Y))+X.shape[0]/2*np.log(2*math.pi)
    return res[0]

def visualization(var, mean):
    upper = np.zeros(points)
    lower = np.zeros(points)
    for i in range(points):
        upper[i] = mean[i, 0] + 1.96*var[i, i]
        lower[i] = mean[i, 0] - 1.96*var[i, i]
    
    plt.xlim(-60, 60)
    plt.scatter(X, Y, s=7.0, c='m')
    plt.plot(Xstar.ravel(), mean.ravel(), 'b')
    plt.fill_between(Xstar.ravel(), upper, lower, alpha=0.5)
    plt.show()

def Gaussian_Process(l=1.0, alpha=1.0):
    C = rational_quadratic_kernel(X, X, l, alpha) + 1/beta*np.identity(X.shape[0])
    k_x_xstar = rational_quadratic_kernel(X, Xstar, l, alpha)
    kstar = rational_quadratic_kernel(Xstar, Xstar, l, alpha)+1/beta

    mean = k_x_xstar.T.dot(np.linalg.inv(C).dot(Y))
    var = kstar - k_x_xstar.T.dot(np.linalg.inv(C).dot(k_x_xstar))

    visualization(var, mean)
    return

if __name__ == '__main__':
    points = 1000
    X, Y = read_data()
    Xstar = np.linspace(-60, 60, points).reshape(-1, 1)

    beta = 5
    # start_time = time.time()
    Gaussian_Process()
    # end_time = time.time()
    # print('part1:',end_time-start_time)
    res = minimize(negative_marginal_likelihood, x0=[1,1])
    # print(res.x[0], res.x[1])
    # start_time = time.time()
    Gaussian_Process(res.x[0], res.x[1])
    # end_time = time.time()
    # print('part2:',end_time-start_time)