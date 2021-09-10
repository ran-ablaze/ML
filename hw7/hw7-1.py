import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def read_dataset(mode = 'Training'):
    path = './Yale_Face_Database/' + mode
    files = os.listdir(path)
    images = np.zeros((len(files), 98*116))
    labels = np.zeros(len(files), dtype=int)
    for i, file in enumerate(files):
        image = Image.open(os.path.join(path, file))
        images[i] = np.asarray(image.resize((98, 116))).flatten()
        labels[i] = int(file[7:9])
    return images, labels

def PCA(train_faces, test_faces, train_label, test_label):
    avg_face = (np.sum(train_faces, axis = 0)/len(train_faces)).flatten()
    diff_train_face = train_faces - avg_face
    S = diff_train_face.T.dot(diff_train_face)
    eigen_values, eigen_vectors = np.linalg.eig(S)
    # np.save('PCA_S.npy',S)
    # np.save('PCA_eigen_values.npy',eigen_values)
    np.save('PCA_eigen_vectors.npy',eigen_vectors)
    # S = np.load('PCA_S.npy')
    # eigen_values = np.load('PCA_eigen_values.npy')
    # eigen_vectors = np.load('PCA_eigen_vectors.npy')

    sort_index = np.argsort(-eigen_values)
    eigen_vectors = eigen_vectors[:,sort_index[0:25]].real
    eigenfaces = eigen_vectors.T
    show_faces(eigenfaces.reshape(25, 116, 98), 5, 'pca eigenfaces')

    chosen_index = random.sample(range(len(test_faces)), 10)
    weight = test_faces[chosen_index].dot(eigen_vectors)
    reconstruction_faces = avg_face + weight.dot(eigenfaces)
    show_faces(reconstruction_faces.reshape(10, 116, 98), 2, 'pca reconstruction faces')

    diff_test_face = test_faces - avg_face
    train_weight = train_faces.dot(eigen_vectors)
    test_weight = test_faces.dot(eigen_vectors)
    face_recognition(train_label, train_weight, test_label, test_weight)
    return

def kernel_PCA(train_faces, test_faces, train_label, test_label):
    avg_face = (np.sum(train_faces, axis = 0)/len(train_faces)).flatten()

    K = RBF_kernel(train_faces.T, train_faces.T)
    l_n = np.ones((98*116,98*116))/(98*116)
    k_prime = K - l_n.dot(K) - K.dot(l_n) + l_n.dot(K.dot(l_n))
    
    eigen_values, eigen_vectors = np.linalg.eig(k_prime)
    # np.save('kernel_PCA_S.npy',k_prime)
    # np.save('kernel_PCA_eigen_values.npy',eigen_values)
    np.save('kernel_PCA_eigen_vectors.npy',eigen_vectors)
    # k_prime = np.load('kernel_PCA_S.npy')
    # eigen_values = np.load('kernel_PCA_eigen_values.npy')
    # eigen_vectors = np.load('kernel_PCA_eigen_vectors.npy')

    sort_index = np.argsort(-eigen_values)
    eigen_vectors = eigen_vectors[:,sort_index[0:25]].real
    eigenfaces = eigen_vectors.T
    show_faces(eigenfaces.reshape(25, 116, 98), 5, 'pca eigenfaces')

    chosen_index = random.sample(range(len(test_faces)), 10)
    weight = test_faces[chosen_index].dot(eigen_vectors)
    reconstruction_faces = avg_face + weight.dot(eigenfaces)
    show_faces(reconstruction_faces.reshape(10, 116, 98), 2, 'pca reconstruction faces')

    diff_test_face = test_faces - avg_face
    train_weight = train_faces.dot(eigen_vectors)
    test_weight = test_faces.dot(eigen_vectors)
    face_recognition(train_label, train_weight, test_label, test_weight)
    return

def LDA(train_faces, test_faces, train_label, test_label):
    avg_face = (np.sum(train_faces, axis = 0)/len(train_faces)).flatten()

    num_of_class = 15
    Sw = np.zeros((98*116, 98*116))
    Sb = np.zeros((98*116, 98*116))
    for i in range(1,num_of_class+1):
        index = np.where(train_label == i)[0]
        faces_i = train_faces[index]
        mean_class_i = (np.sum(faces_i, axis = 0)/len(faces_i)).flatten()
        diff_class_i = faces_i-mean_class_i
        diff_class = mean_class_i - avg_face

        Sw += diff_class_i.T.dot(diff_class_i)
        Sb += len(index) * diff_class.T.dot(diff_class)
    
    S = np.linalg.inv(Sw).dot(Sb)
    eigen_values, eigen_vectors = np.linalg.eig(S)
    # np.save('LDA_S.npy',S)
    # np.save('LDA_eigen_values.npy',eigen_values)
    # np.save('LDA_eigen_vectors.npy',eigen_vectors)
    # S = np.load('LDA_S.npy')
    # eigen_values = np.load('LDA_eigen_values.npy')
    # eigen_vectors = np.load('LDA_eigen_vectors.npy')
    eigen_vectors_pca = np.load('PCA_eigen_vectors.npy').real

    sort_index = np.argsort(-eigen_values)
    eigen_vectors = eigen_vectors[:,sort_index[0:25]].real
    fisherfaces =  eigen_vectors_pca.dot(eigen_vectors).T
    show_faces(fisherfaces.reshape(25, 116, 98), 5, 'lda fisherfaces')

    chosen_index = random.sample(range(len(test_faces)), 10)
    weight = test_faces[chosen_index].dot(fisherfaces.T)
    reconstruction_faces = avg_face + weight.dot(fisherfaces)
    show_faces(reconstruction_faces.reshape(10, 116, 98), 2, 'lda reconstruction faces')

    diff_test_face = test_faces - avg_face
    train_weight = train_faces.dot(fisherfaces.T)
    test_weight = test_faces.dot(fisherfaces.T)
    face_recognition(train_label, train_weight, test_label, test_weight)
    return

def kernel_LDA(train_faces, test_faces, train_label, test_label):
    avg_face = (np.sum(train_faces, axis = 0)/len(train_faces)).flatten()
    K = RBF_kernel(train_faces.T, train_faces.T)
    M_star = (np.sum(K, axis = 0)/len(train_faces)).flatten()
    num_of_class = 15
    N = np.zeros((98*116, 98*116))
    M = np.zeros((98*116, 98*116))
    l_lj = np.ones((9,9))/9
    for i in range(1,num_of_class+1):
        index = np.where(train_label == i)[0]
        Kj = K[index]
        lj = len(index)
        Mj = (np.sum(Kj, axis = 0)/len(Kj)).flatten()
        diff_class = Mj - M_star

        N += Kj.T.dot((np.identity(lj)-l_lj).dot(Kj))
        M += lj * diff_class.T.dot(diff_class)
    S = np.linalg.pinv(N).dot(M)
    eigen_values, eigen_vectors = np.linalg.eigh(S)
    # np.save('kernel_LDA_S.npy',S)
    # np.save('kernel_LDA_eigen_values.npy',eigen_values)
    # np.save('kernel_LDA_eigen_vectors.npy',eigen_vectors)
    # S = np.load('kernel_LDA_S.npy')
    # eigen_values = np.load('kernel_LDA_eigen_values.npy')
    # eigen_vectors = np.load('kernel_LDA_eigen_vectors.npy')
    eigen_vectors_pca = np.load('kernel_PCA_eigen_vectors.npy').real

    sort_index = np.argsort(-eigen_values)
    eigen_vectors = eigen_vectors[:,sort_index[0:25]].real
    fisherfaces =  eigen_vectors_pca.dot(eigen_vectors).T
    show_faces(fisherfaces.reshape(25, 116, 98), 5, 'lda fisherfaces')

    chosen_index = random.sample(range(len(test_faces)), 10)
    weight = test_faces[chosen_index].dot(fisherfaces.T)
    reconstruction_faces = avg_face + weight.dot(fisherfaces)
    show_faces(reconstruction_faces.reshape(10, 116, 98), 2, 'lda reconstruction faces')

    diff_test_face = test_faces - avg_face
    train_weight = train_faces.dot(fisherfaces.T)
    test_weight = test_faces.dot(fisherfaces.T)
    face_recognition(train_label, train_weight, test_label, test_weight)
    return

def RBF_kernel(xi, xj, gamma=0.0001):
    return np.exp( -gamma * cdist(xi, xj,'sqeuclidean'))

def show_faces(faces, row, title):
    plt.figure(title)
    for idx in range(len(faces)):
        plt.subplot(row, int((len(faces)+1)/row), idx+1)
        plt.axis('off')
        plt.imshow(faces[idx], cmap='gray')
    plt.show()

def face_recognition(train_label, train_weight, test_label, test_weight, k=5):
    error = 0
    dist = np.zeros(len(train_weight))
    for i in range(len(test_weight)):
        for j in range(len(train_weight)):
            dist[j] = cdist([test_weight[i]], [train_weight[j]], 'sqeuclidean')
        k_nearst = np.argsort(dist)[0:k]
        predict = np.argmax(np.bincount(train_label[k_nearst]))
        if test_label[i] != predict:
            error += 1
    print("Accuracy:", (len(test_label)-error)/len(test_label))

if __name__ == '__main__':
    train_image, train_label = read_dataset(mode = 'Training')
    test_image, test_label = read_dataset(mode = 'Testing')
    
    PCA(train_image, test_image, train_label, test_label)
    # LDA(train_image, test_image, train_label, test_label)
    # kernel_PCA(train_image, test_image, train_label, test_label)
    # kernel_LDA(train_image, test_image, train_label, test_label)