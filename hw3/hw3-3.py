import numpy as np
import argparse
import matplotlib.pyplot as plt
from hw3_1 import Polynomial_basis_linear_model

parser = argparse.ArgumentParser()
parser.add_argument('--b', type=float, default=0.0)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--w', nargs='+', type=float, default=1.0)
args = parser.parse_args()

def design_matrix(x, n):
    X = np.zeros((1, n))
    for i in range(n):
        X[0][i] = x**i
    return X

def Baysian_Linear_regression():
    data_x = []
    data_y = []
    data_mean = []
    data_var = []
    prior_cov, posterior_cov =  np.identity(n), np.zeros((n, n))
    prior_mean, posterior_mean = np.zeros((n, 1)),  np.zeros((n, 1))
    prior_var, posterior_var = np.zeros((n, n)), np.zeros((n, n))
    iteration = 0

    while True:
        x, y = Polynomial_basis_linear_model(n, a, w)
        X = design_matrix(x, n)
        data_x.append(x)
        data_y.append(y)
        print('Add data point ({}, {}):'.format(x, y))
        print()

        if iteration == 0:
            #Λ=aXTX+bI, μ=aΛ^-1XTY
            posterior_cov = a * np.matmul(X.transpose(), X) + b * np.identity(n)
            posterior_mean = a * np.matmul(np.linalg.inv(posterior_cov), X.transpose()) * y
        else:
            #Λ=aXTX+S, μ=Λ^-1(aXTY+Sm)
            posterior_cov = a * np.matmul(X.transpose(), X) + prior_cov
            posterior_mean = np.matmul(np.linalg.inv(posterior_cov), a * X.transpose() * y + np.matmul(prior_cov, prior_mean))

        print('Postirior mean:')
        for i in range(n):
            print('{:15.10f}'.format(posterior_mean[i,0]))
        print()

        print('Postirior variance:')
        posterior_var = np.linalg.inv(posterior_cov)
        for i in range(n):
            for j in range(n):
                if j < n-1:
                    print('{:15.10f}'.format(posterior_var[i,j]), end=',')
                else:
                    print('{:15.10f}'.format(posterior_var[i,j]))
        print()

        #N(Xm, 1/a+XS^-1XT)
        predict_mean = np.matmul(X, prior_mean)[0][0]
        predict_var = (1 / a + np.matmul(X, np.matmul(np.linalg.inv(prior_cov), X.transpose())))[0][0]
        print('Predictive distribution ~ N({:.5f}, {:.5f})'.format(predict_mean, predict_var))
        print('--------------------------------------------------')

        if iteration == 10 or iteration == 50:
            data_var.append(posterior_var)
            data_mean.append(posterior_mean)
        
        if (abs(posterior_mean-prior_mean)<0.00001).all() and iteration > 50:
            data_var.append(posterior_var)
            data_mean.append(posterior_mean)
            break
        prior_cov = posterior_cov
        prior_mean = posterior_mean
        prior_var = posterior_var
        iteration += 1

    visualization(data_x, data_y, data_mean, data_var)
    return

def visualization(data_x, data_y, data_mean, data_var, num_point=50):
    x = np.linspace(-2.0, 2.0, num_point)
    X = []
    for i in range(num_point):
        X.append(design_matrix(x[i], n))

    plt.subplot(221)
    plt.title("Ground truth")
    func = np.poly1d(np.flip(w))
    y = func(x)
    var = a
    draw(x, y, var)
        
    plt.subplot(222)
    plt.title("Predict result")
    func = np.poly1d(np.flip(np.reshape(data_mean[2], n)))
    y = func(x)
    var = np.zeros((num_point))
    for i in range(num_point):
        var[i] = a +  X[i].dot(data_var[2].dot(X[i].T))[0][0]
    plt.scatter(data_x, data_y, s=7.0, alpha=0.5)
    draw(x, y, var)

    plt.subplot(223)
    plt.title("After 10 incomes")
    func = np.poly1d(np.flip(np.reshape(data_mean[0], n)))
    y = func(x)
    for i in range(num_point):
        var[i] = a +  X[i].dot(data_var[0].dot(X[i].T))[0][0]
    plt.scatter(data_x[0:10], data_y[0:10], s=7.0, alpha=0.5)
    draw(x, y, var)

    plt.subplot(224)
    plt.title("After 50 incomes")
    func = np.poly1d(np.flip(np.reshape(data_mean[1], n)))
    y = func(x)
    for i in range(num_point):
        var[i] = a +  X[i].dot(data_var[1].dot(X[i].T))[0][0]
    plt.scatter(data_x[0:50], data_y[0:50], s=7.0, alpha=0.5)
    draw(x, y, var)

    plt.tight_layout()
    plt.show()

def draw(x, y, var):
	plt.plot(x, y, color = 'black')
	plt.plot(x, y+var, color = 'red')
	plt.plot(x, y-var, color = 'red')
	plt.xlim(-2.0, 2.0)
	plt.ylim(-20.0, 30.0)

if __name__ == '__main__':
    b = args.b
    n = args.n
    a = args.a
    w = args.w

    Baysian_Linear_regression()
