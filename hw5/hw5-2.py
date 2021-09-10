import sys
import numpy as np
from libsvm.svmutil import *
import time
from scipy.spatial.distance import cdist

def read_dataset(train = True):
    if train == True:
        image_path = 'X_train.csv'
        label_path = 'Y_train.csv'
    else:
        image_path = 'X_test.csv'
        label_path = 'Y_test.csv'
    
    images = np.loadtxt(image_path, dtype=np.float, delimiter=',')
    labels = np.loadtxt(label_path, dtype=np.int, delimiter=',')

    return images, labels

def svm(kernel_type):
    # -t kernel_type : set type of kernel function (default 2)
	# 0 -- linear: u'*v
	# 1 -- polynomial: (gamma*u'*v + coef0)^degree
	# 2 -- radial basis function: exp(-gamma*|u-v|^2)
    print('kernel_type:', end='')
    if kernel_type == 0:
        print('linear')
    elif kernel_type == 1:
        print('polynomial')
    elif kernel_type == 2:
        print('RBF')
    else:
        print('Error kernel type!')
        exit(-1)
    
    time_start = time.time()
    param = svm_parameter('-t '+str(kernel_type)+' -q')
    prob  = svm_problem(Y_train, X_train)
    m = svm_train(prob, param)
    _, p_acc, _ = svm_predict(Y_test, X_test, m)
    time_end = time.time()
    print("Total time: %0.2f seconds." % (time_end-time_start))
    print()

def svm_Grid_search(kernel_type):
    # -t kernel_type : set type of kernel function (default 2)
	# 0 -- linear: u'*v
	# 1 -- polynomial: (gamma*u'*v + coef0)^degree
	# 2 -- radial basis function: exp(-gamma*|u-v|^2)
    n = 3

    print('kernel_type:', end='')
    time_start = time.time()
    max_acc = 0
    if kernel_type == 0:
        print('linear')
        for c in [0.001, 0.01, 0.1, 1, 10, 100]:
            param = svm_parameter('-t '+str(kernel_type)+' -c '+str(c)+' -v '+str(n)+' -q')
            prob  = svm_problem(Y_train, X_train)
            sys.stdout = None
            p_acc = svm_train(prob, param)
            sys.stdout = sys.__stdout__
            if p_acc > max_acc:
                max_acc = p_acc
                best_params = {'C':c}
    elif kernel_type == 1:
        print('polynomial')
        for degree in range(0, 5):
            for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
                for coef0 in range(-5, 5, 1):
                    for c in [0.001, 0.01, 0.1, 1, 10, 100]:
                        param = svm_parameter('-t '+str(kernel_type)+
                            ' -d '+str(degree)+' -g '+str(gamma)+' -r '+str(coef0)+
                            ' -c '+str(c)+' -v '+str(n)+' -q')
                        prob  = svm_problem(Y_train, X_train)
                        sys.stdout = None
                        p_acc = svm_train(prob, param)
                        sys.stdout = sys.__stdout__
                        if p_acc > max_acc:
                            max_acc = p_acc
                            best_params = {'degree':degree,'gamma':gamma,'coef0':coef0,'C':c}
    elif kernel_type == 2:
        print('RBF')
        for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
            for c in [0.001, 0.01, 0.1, 1, 10, 100]:
                param = svm_parameter('-t '+str(kernel_type)+
                    ' -g '+str(gamma)+' -c '+str(c)+' -v '+str(n)+' -q')
                prob  = svm_problem(Y_train, X_train)
                sys.stdout = None
                p_acc = svm_train(prob, param)
                sys.stdout = sys.__stdout__
                if p_acc > max_acc:
                    max_acc = p_acc
                    best_params = {'gamma':gamma,'C':c}
    else:
        print('Error kernel type!')
        exit(-1)
    time_end = time.time()
    print("Max accuracy", max_acc)
    print("Best Parameters:", best_params)
    print("Total time: %0.2f seconds." % (time_end-time_start))
    print()

def linear_kernel(u, v):
    return u.dot(v.T)

def RBF_kernel(u, v, gamma = 1/784):
    return np.exp(-gamma * cdist(u, v, 'sqeuclidean'))

def svm_precomputed_kernel():
    time_start = time.time()
    X_train_new = linear_kernel(X_train, X_train)+RBF_kernel(X_train, X_train)
    X_test_new = linear_kernel(X_test, X_test)+RBF_kernel(X_test, X_test)
    X_train_new = np.hstack((np.arange(1, 5000+1).reshape(-1, 1), X_train_new))
    X_test_new = np.hstack((np.arange(1, 2500+1).reshape(-1, 1), X_test_new))
    param = svm_parameter('-t 4 -q')
    prob  = svm_problem(Y_train, X_train_new, isKernel=True)
    m = svm_train(prob, param)
    _, p_acc, _ = svm_predict(Y_test, X_test_new, m)
    time_end = time.time()
    print("Total time: %0.2f seconds." % (time_end-time_start))
    print()

if __name__ == '__main__':
    X_train, Y_train = read_dataset(train=True)
    X_test, Y_test = read_dataset(train=False)

    print('******* Part1 *******')
    svm(0)    #linear
    svm(1)    #polynomial
    svm(2)    #RBF

    print('******* Part2 *******')
    svm_Grid_search(0)
    svm_Grid_search(1)
    svm_Grid_search(2)

    print('******* Part3 *******') 
    svm_precomputed_kernel()