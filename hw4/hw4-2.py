import os
import numpy as np
import struct
from numba import jit

NUM_CLASSES = 10
PI = 3.141592653589793

def read_dataset(root, train = True):
    if train == True:
        image_path = os.path.join(root, 'train-images.idx3-ubyte')
        label_path = os.path.join(root, 'train-labels.idx1-ubyte')
    else:
        image_path = os.path.join(root, 't10k-images.idx3-ubyte')
        label_path = os.path.join(root, 't10k-labels.idx1-ubyte')
    
    with open(image_path, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = np.fromfile(file, dtype=np.dtype('B'), count=-1)
        images = image_data.reshape(size, rows*cols)

    with open(label_path, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = np.fromfile(file, dtype=np.dtype('B'), count=-1)             

    return images, labels

@jit
def transform_data(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i,j] > 127):
                image[i,j] = 1
            else:
                image[i,j] = 0
    return image

@jit
def E_step(P, Lambda, X, w):
    for i in range(train_num):
        for j in range(NUM_CLASSES):
            w[i,j] = Lambda[j]
            for k in range (pixels):
                if X[i, k]:
                    w[i,j] *= P[j,k]
                else:
                    w[i,j] *= (1-P[j,k])
        if np.sum(w[i,:]):
            w[i,:] /= np.sum(w[i,:])
    return w

@jit
def M_step(w, X, P, Lambda):
    Lambda = np.sum(w, axis=0)
    for i in range(NUM_CLASSES):
        for j in range(pixels):
            P[i,j] = 0
            for k in range(train_num):
                P[i,j] += w[k,i]*X[k,j]
            P[i,j] = (P[i,j]+1e-8)/(Lambda[i]+1e-8*784)
        Lambda[i] = (Lambda[i]+1e-8)/(np.sum(Lambda)+1e-8*10)
    return P, Lambda

def print_imagination(P, iter, diff):
    Class = np.zeros((NUM_CLASSES, pixels))
    Class = (P>=0.5)*1
    for i in range(NUM_CLASSES):
        print("class {}:".format(i))
        for m in range (28):
            for n in range(28):
                print(Class[i][m*28+n], end=' ')
            print()
        print()

    print("No. of Iteration: {}, Difference: {}".format(iter, diff))
    print("-----------------------------------------\n")
    
def print_label_imagination(P, label):
    Class = np.zeros((NUM_CLASSES, pixels))
    Class = (P>=0.5)*1
    for i in range(NUM_CLASSES):
        idx = label[i]
        print("labeled class {}:".format(i))
        for m in range (28):
            for n in range(28):
                print(Class[idx][m*28+n], end=' ')
            print()
        print()

    print()

@jit
def predict_label():
    P_label = np.zeros(10)
    label_predict_num = np.zeros((10, 10))

    for i in range(train_num):
        for j in range(NUM_CLASSES):
            P_label[j] = Lambda[j]
            for k in range (pixels):
                if X[i,k]:
                    P_label[j] *= P[j,k]
                else:
                    P_label[j] *= (1-P[j,k])
        label_predict_num[train_label[i], np.argmax(P_label)] += 1
    return label_predict_num

def assign_label(label_predict_num):
    label_class = np.full(10, -1)
    class_label = np.full(10, -1)
    P_label = np.zeros((10, 10))
    for i in range(NUM_CLASSES):
        P_label[i,:] = label_predict_num[i,:]/np.sum(label_predict_num[i,:])
    P_label = P_label.ravel()
    
    i = 0
    while i < 10:
        temp = np.argmax(P_label)
        if P_label[temp] == 0:
            break
        P_label[temp] = 0
        if label_class[(int)(temp/10)] == -1 and class_label[temp%10] == -1:
            label_class[(int)(temp/10)] = temp%10
            class_label[temp%10] = (int)(temp/10)
            i+=1
    return label_class

def print_confusion_mat(Lambda, confusion, label):
    error = 60000
    for i in range(NUM_CLASSES):
        tp = confusion[i, label[i]]
        fp = np.sum(confusion[i])-tp
        fn = np.sum(confusion[:,label[i]])-tp
        tn = 60000-tp-fp-fn
        error -= tp

        print("-----------------------------------------\n")
        print('Confusion Matrix {}:'.format(i))
        print('{:^20}{:^25}{:^25}'.format(' ', 'Predict number %d'%i, 'Predict not number %d'%i))
        print('{:^20}{:^25}{:^25}'.format('Is number %d'%i, tp, fn))
        print('{:^20}{:^25}{:^25}\n'.format('Isn\'t number %d'%i, fp, tn))
        print('Sensitivity (Successfully predict number {}):     {}'.format(i, tp/(tp+fn)))
        print('Specificity (Successfully predict not number {}): {}'.format(i, tn/(fp+tn))) #fp/(fp+tn)
    print('Total iteration to converge:', iter)
    print('Total error rate:', error/60000)

    
if __name__ == '__main__':
    train_image, train_label = read_dataset('./data', train=True)
    test_image, test_label = read_dataset('./data', train=False)

    train_num = train_image.shape[0]
    pixels = train_image.shape[1]
    X = transform_data(train_image)
    P = np.random.uniform(0.0, 1.0, (NUM_CLASSES,pixels))
    for i in range(NUM_CLASSES):
        P[i,:] /= np.sum(P[i,:])
    Lambda = np.full(NUM_CLASSES, 0.1)
    w = np.zeros((train_num, NUM_CLASSES))
    # np.save('./npy/p.npy', P)
    # np.save('./npy/lambda.npy', Lambda)
    iter = 0

    while True:
        P_prev = np.copy(P)
        iter += 1
        w = E_step(P, Lambda, X, w)
        np.save('./npy/w'+str(iter)+'.npy', w)
        # w = np.load('./npy/w.npy')

        P, Lambda = M_step(w, X, P, Lambda)
        np.save('./npy/p'+str(iter)+'.npy', P)
        np.save('./npy/lambda'+str(iter)+'.npy', Lambda)
        # P = np.load('/npy/p'+iter+'.npy')
        # Lambda = np.load('./npy/lambda'+iter+'.npy')
        diff = np.linalg.norm(P-P_prev)
        print_imagination(P, iter, diff)
        if iter==20 or diff<1e-2:
            break
    # w = np.load('./npy/w'+str(iter)+'.npy')
    # P = np.load('./npy/p'+str(iter)+'.npy')
    # Lambda = np.load('./npy/lambda'+str(iter)+'.npy')

    confusion = predict_label()
    label = assign_label(confusion)
    print_label_imagination(P, label)
    print_confusion_mat(Lambda, confusion, label)
