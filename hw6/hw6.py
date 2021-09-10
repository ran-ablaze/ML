import os
import numpy as np
import argparse
import random
from PIL import Image
from scipy.spatial.distance import pdist, cdist
import imageio
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--input', type=str, default='image1.png')
parser.add_argument('--output', type=str, default='output')
parser.add_argument('--s', type=float, default=0.001)
parser.add_argument('--c', type=float, default=0.01)
args = parser.parse_args()

def read_image():
    image = Image.open(args.input)
    data = np.array(image.getdata())
    return data 

def write_image(cluster, model, num):
    colors = np.array([[255,0,0],[0,255,0],[0,0,255],[0,215,175],[95,0,135],[255,255,0],[255,175,0]])
    data = np.zeros((10000, 3))
    for i in range(10000):
        data[i,:] = colors[cluster[i],:]
    
    image = data.reshape(100, 100, 3)
    image = Image.fromarray(np.uint8(image))
    image.save(os.path.join(args.output, model+'%d.png'%num))
    return

def compose_gif(model, iter):
    gif_images = []
    for i in range(iter):
        gif_images.append(imageio.imread(os.path.join(args.output, model+'%d.png'%i)))
    imageio.mimsave(model+'%d.gif'%args.k,gif_images,fps=1)

def kernel(x, gamma_s=args.s, gamma_c=args.c):
    dist_c = cdist(x, x, 'sqeuclidean')

    grid = np.indices((100,100)).reshape(2,10000,1)
    S_x = np.hstack((grid[0], grid[1]))
    dist_s = cdist(S_x, S_x, 'sqeuclidean')

    kernel = np.multiply(np.exp(-gamma_s * dist_s), np.exp(-gamma_c * dist_c))
    return kernel

def cal_distance(x, y):
    return kernel[x,x]+kernel[y,y]-2*kernel[x,y]

def kernel_k_means(data, K=args.k):
    # centroids = list(random.sample(range(0,10000), K))
    centroids = []
    centroids = list(random.sample(range(0,10000), 1))
    for number_center in range(1, K):
        min_dist = np.full(10000, np.inf)
        for i in range(10000):
            for j in range(number_center):
                dist = cal_distance(i, centroids[j])
                if dist < min_dist[i]:
                    min_dist[i] = dist
        min_dist /= np.sum(min_dist)
        centroids.append(np.random.choice(np.arange(10000), 1, p=min_dist)[0])
    
    cluster = np.zeros(10000, dtype=int)
    for i in range(10000):
        dist = np.full(K, np.inf)
        for j in range(K):
            dist[j] = cal_distance(i,centroids[j])
        cluster[i] = np.argmin(dist)
    write_image(cluster, 'kmeans', 0)

    for i in range(1, 10):
        print("iter", i)
        prev_cluster = cluster
        cluster = np.zeros(10000, dtype=int)

        _, C = np.unique(prev_cluster, return_counts=True)
        k_pq = np.zeros(K)
        for k in range(K):
            temp = kernel.copy()
            for n in range(10000):
                if prev_cluster[n]!=k:
                    temp[n,:] = 0
                    temp[:,n] = 0
            k_pq[k] = np.sum(temp)

        for j in range(10000):
            dist = np.full(K, np.inf)
            for k in range(K):
                temp = kernel[j,:].copy()
                index = np.where(prev_cluster == k)
                k_jn = np.sum(temp[index])
                
                dist[k] = kernel[j,j]-2/C[k]*k_jn+(1/C[k]**2)*k_pq[k]
            cluster[j] = np.argmin(dist)
        
        if(np.linalg.norm((cluster-prev_cluster), ord=2)<1e-2):
            break
        write_image(cluster, 'kmeans', i)
    return cluster, i

def spectral_clustering_ratio(K=args.k):
    D, L = Laplacian(kernel)
    # eigenvalue, eigenvector = np.linalg.eig(L)
    # eigenvector = eigenvector.T
    # np.save("ratio_eigenvalue.npy", eigenvalue)
    # np.save("ratio_eigenvector.npy", eigenvector)
    eigenvalue = np.load("ratio_eigenvalue.npy")
    eigenvector = np.load("ratio_eigenvector.npy")

    sort_idx = np.argsort(eigenvalue)
    mask = eigenvalue[sort_idx] > 0
    idx = sort_idx[mask][0:args.k]
    U = eigenvector[idx].T

    cluster, i = kmeans(U, K)
    show_eigen(U, cluster)
    return cluster, i

def spectral_clustering_normalized(K=args.k):
    D, L = Laplacian(kernel)
    Dsym = np.zeros((D.shape))
    for i in range(len(D)):
        Dsym[i,i] = D[i,i]**-0.5
    Lsym = Dsym.dot(L.dot(Dsym))
    # eigenvalue, eigenvector = np.linalg.eig(Lsym)
    # eigenvector = eigenvector.T
    # np.save("normalized_eigenvalue.npy", eigenvalue)
    # np.save("normalized_eigenvector.npy", eigenvector)
    eigenvalue = np.load("normalized_eigenvalue.npy")
    eigenvector = np.load("normalized_eigenvector.npy")

    sort_idx = np.argsort(eigenvalue)
    mask = eigenvalue[sort_idx] > 0
    idx = sort_idx[mask][0:args.k]
    U = eigenvector[idx].T
    T = U.copy()
    temp = np.sum(U, axis=1)
    for i in range(len(T)):
        T[i] /= temp[i]

    cluster, i = kmeans(T, K, mode = 1)
    show_eigen(U, cluster, mode = 'normalized')
    return cluster, i

def Laplacian(W):
    D = np.zeros((W.shape))
    L = np.zeros((W.shape))
    for r in range(len(W)):
        for c in range(len(W)):
            D[r,r] += W[r,c]
    L = D-W
    return D, L

def kmeans(data, k, mode = 0):
    if mode == 0:
        title = 'ratio'
    else:
        title = 'normalized'
    # centroids = list(random.sample(range(0,10000), k))
    centroids = []
    centroids = list(random.sample(range(0,10000), 1))
    for number_center in range(1, k):
        min_dist = np.full(10000, np.inf)
        for i in range(10000):
            for j in range(number_center):
                dist = cdist([data[i]], [data[centroids[j]]], 'sqeuclidean')
                if dist < min_dist[i]:
                    min_dist[i] = dist
        min_dist /= np.sum(min_dist)
        centroids.append(np.random.choice(np.arange(10000), 1, p=min_dist)[0])
    centers = []
    for i in range(k):
        centers.append(data[centroids[i]])
    centers = np.array(centers)

    cluster = clustering(data, centers, k)
    write_image(cluster, title, 0)

    for i in range(1, 10):
        print("iter ", i)
        prev_centers = centers
        centers = []
        for j in range(k):
            mask = cluster==j
            centers.append(np.sum(data[mask], axis=0) / len(data[mask]))
        centers = np.array(centers)
        if(np.linalg.norm((centers-prev_centers), ord=2)<1e-2):
            break

        cluster = clustering(data, centers, k)
        write_image(cluster, title, i)
    return cluster, i

def clustering(U, centers, k):
    cluster = np.zeros(10000, dtype=int)
    for i in range(10000):
        dist = np.full(k, np.inf)
        for j in range(k):
            dist[j] = cdist([U[i]], [centers[j]], 'sqeuclidean')
        cluster[i] = np.argmin(dist)
    return cluster

def show_eigen(data, cluster, mode='ratio'):
    colors = np.array([[255,0,0],[0,255,0],[0,0,255],[0,215,175],[95,0,135],[255,255,0],[255,175,0]])
    plt.clf()
    for idx in range(len(data)):
        plt.scatter(data[idx,0], data[idx,1], c= colors[cluster[idx]])
    plt.savefig("eigen_"+mode+".png")
    plt.show()
    return

if __name__=='__main__':
    data = read_image() #(10000,3)
    kernel = kernel(data)

    ### kmeans
    cluster, iter = kernel_k_means(kernel)
    compose_gif('kmeans', iter)

    ###
    cluster, iter = spectral_clustering_ratio()
    compose_gif('ratio', iter)
    cluster, iter = spectral_clustering_normalized()
    compose_gif('normalized', iter)

