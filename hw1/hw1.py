import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='testfile.txt')
parser.add_argument('--n', type=int, required=True)
parser.add_argument('--lamb', type=float, required=True)
args = parser.parse_args()

def read_data(file):
    data_x=[]; data_y=[]
    fp = open(file, "r")
    line = fp.readline()
    while line:
        x, y = list(map(float, line.strip().split(',')))
        data_x.append(x)
        data_y.append(y)
        line = fp.readline()
    fp.close()
    return data_x, data_y

def LSE(A, B):
    ATA_p_lambI= matmul(np.transpose(A),A)+args.lamb*I(len(A[0]))
    ATB = matmul(np.transpose(A),B)
    L, U = LU_decomposition(ATA_p_lambI)
    x = LU_inverse(L, U, ATB)
    return x

def Newtons_method(A, B):
    Hessian_f = 2*matmul(np.transpose(A),A)
    # Hessian_f = 2*ATA
    Hessian_L, Hessian_U = LU_decomposition(Hessian_f)
    ATB_2 = 2*matmul(np.transpose(A),B)

    x = np.random.rand(n, 1)
    Gradient_f = matmul(Hessian_f, x) - ATB_2
    # x = x0 - HF^-1 * gradient_F
    x = x - LU_inverse(Hessian_L, Hessian_U, Gradient_f)#Hessian_f_inv, Gradient_f)
    return x

def print_result(x):
    print("Fitting line:", end=' ')
    for i in range(len(x)):
        if x[i]==0.0:
            continue
        if i!=0 and x[i]>0:
            print("+", end=' ')
        print("%.12g"%x[i][0], end='')
        if i<n-1:
            print("X^"+str(n-i-1), end=' ')
    print()

    print("Total error:", end=' ')
    Ax_m_b = matmul(A,x) - B
    error = matmul(np.transpose(Ax_m_b),Ax_m_b)
    print("%.11g"%error)
    return

def visualization(w_LSE, w_Newton):
    minx = min(data_x)
    maxx = max(data_x)
    miny = min(data_y)
    maxy = max(data_y)
    t = np.arange(minx-2, maxx+1+2)

    y_LSE=0
    y_Newton=0
    for i in range(n):
        y_LSE += w_LSE[i]*(t**(n-i-1))
        y_Newton += w_Newton[i]*(t**(n-i-1))

    plt.figure(1)

    plt.subplot(211)
    plt.scatter(data_x, data_y, c='r', edgecolors='k')
    plt.plot(t,y_LSE, c='k')
    plt.xlim(minx-(maxx-minx)/10, maxx+(maxx-minx)/10)
    plt.ylim(miny-(maxy-miny)/10, maxy+(maxy-miny)/10)
    
    plt.subplot(212)
    plt.scatter(data_x, data_y, c='r', edgecolors='k')
    plt.plot(t,y_Newton, c='k')
    plt.xlim(minx-(maxx-minx)/10, maxx+(maxx-minx)/10)
    plt.ylim(miny-(maxy-miny)/10, maxy+(maxy-miny)/10)

    plt.show()
    return

def matmul(A, B):
    row = len(A)
    col = len(B[0])
    col_A = len(A[0])
    assert col_A==len(B)

    C = np.zeros((row, col))
    for k in range(col_A):
        for i in range(row):
            for j in range(col):
                C[i,j]=C[i,j]+A[i,k]*B[k,j]
    
    return C

def I(row):
    M = np.zeros((row, row))
    for i in range(row):
        M[i,i] = 1.0
    return M

def LU_decomposition(M):
    row = len(M)
    U = np.zeros((row, row))
    for i in range(row):
        for j in range(row):
            U[i,j] = M[i,j]
    L = np.zeros((row, row))
    for i in range(row):
        L[i,i] = 1.0

    for i in range(row):
        for j in range(i+1,row):
            L[j,i]=U[j,i]/U[i,i]
            for k in range(i,row):
                U[j,k]=U[j,k]-L[j,i]*U[i,k]
    return L,U

def LU_inverse(L, U, b):
    #Ax=b => L(Ux)=b
    #Ly=b
    row = len(L)
    col = len(b[0])
    y = np.zeros((row, col))
    for j in range(col):
        for i in range(row):
            y[i][j] = b[i][j]
            for k in range(i):
                y[i][j] -= L[i][k]*y[k][j]
    #Ux = y
    x = np.zeros((row, col))
    for j in range(col):
        for i in range(row-1, -1, -1):
            x[i][j] = y[i][j]
            for k in range(row-1, i, -1):
                x[i][j] -= U[i][k]*x[k][j]
            x[i][j] /= U[i][i]
    return x

if __name__ == '__main__':
    n = args.n
    data_x, data_y = read_data(args.file)
    A=[]; B=[]
    for i in range(len(data_x)):
        xi = []
        for j in range(n-1,-1,-1):
            xi.append(data_x[i]**j)
        A.append(xi)
    A = np.asarray(A)
    B = np.asarray(data_y)
    B = np.transpose([B])

    w_LSE = LSE(A,B)
    print("LSE:")
    print_result(w_LSE)
    # Newtons_method(A,B)
    w_NtM = Newtons_method(A,B)
    print("Newton's Method:")
    print_result(w_NtM)
    visualization(w_LSE, w_NtM)
