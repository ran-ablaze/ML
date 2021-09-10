import os
import argparse
import numpy as np
import struct

NUM_CLASSES = 10
NUM_BINS = 32
PI = 3.141592653589793
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=0,
                    help='0: discrete mode 1: continuous mode')
args = parser.parse_args()

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

def cal_prior_prob(labels):
    total_num = len(labels)
    prior = np.zeros(NUM_CLASSES, dtype=float)
    for value in labels:
        prior[value] += 1
    prior /= total_num

    return prior

def cal_likelihood(images, labels):
    pixels = len(images[0])
    likelihood = np.zeros((NUM_CLASSES, pixels, NUM_BINS), dtype=float)
    for i in range(len(images)):
        for j in range(pixels):
            bin = images[i][j]//8
            likelihood[labels[i]][j][bin] += 1 

    total_num = np.sum(likelihood, axis=2)
    for i in range(NUM_CLASSES):
        for j in range(pixels):
            likelihood[i][j][:] /= total_num[i][j]

    #pesudocount
    for i in range(NUM_CLASSES):
        for j in range(pixels):
            for k in range(NUM_BINS):
                if likelihood[i][j][k] == 0:
                    likelihood[i][j][k] = 0.0002
    return likelihood

def cal_mean_var(images, labels, prior):
    pixels = len(images[0])
    mean = np.zeros((NUM_CLASSES, pixels), dtype=float)
    var = np.zeros((NUM_CLASSES, pixels), dtype=float)

    for i in range(len(images)):
        for j in range(pixels):
            mean[labels[i]][j] += images[i][j]
    
    for i in range(NUM_CLASSES):
        #prior[i]*len(images)=num of class i
        mean[i][:] /= prior[i]*len(images)

    for i in range(len(images)):
        for j in range(pixels):
            var[labels[i]][j] += (images[i][j]-mean[labels[i]][j])**2
    
    for i in range(NUM_CLASSES):
        #prior[i]*len(images)=num of class i
        var[i][:] /= prior[i]*len(images)
    
    return mean, var

def predict_discrete_result(images, labels, likelihood, prior):
    error = 0
    pixels = len(images[0])
    for i in range(len(images)):
        posterior = np.zeros(NUM_CLASSES, dtype=float)
        for n in range (NUM_CLASSES):
            for j in range(pixels):
                bin = images[i][j]//8
                posterior[n] += np.log(likelihood[n][j][bin])
            posterior[n] += np.log(prior[n])
        
        posterior = np.divide(posterior, sum(posterior))
        print('Postirior (in log scale):')
        for n in range(NUM_CLASSES):
            print(n,':',posterior[n])
        pred = np.argmin(posterior)
        print('Prediction:', pred,'Ans:',labels[i])
        print()
        if pred != labels[i]:
            error += 1
    
    error_rate = error/len(images)
    return error_rate

def predict_continuous_result(images, labels, mean, var, prior):
    error = 0
    pixels = len(images[0])

    for i in range(len(images)):
        posterior = np.zeros(10, dtype=float)
        for n in range (NUM_CLASSES):
            for j in range(pixels):
                if var[n][j] == 0:
                    continue
                posterior[n] -= np.log(2.0*PI*var[n][j])/2.0
                posterior[n] -= ((images[i][j]-mean[n][j])**2)/(2.0*var[n][j])
            posterior[n] += np.log(prior[n])
        posterior = np.divide(posterior, sum(posterior))
        print('Postirior (in log scale):')
        for n in range(NUM_CLASSES):
            print("{}: {:.17f}".format(n, posterior[n]))
            # print(n,':',posterior[n])
        pred = np.argmin(posterior)
        print('Prediction: ', pred,'Ans:',labels[i])
        print()
        if pred != labels[i]:
            error += 1
    
    error_rate = error/len(images)
    return error_rate

def show_discrete_imagination(likelihood):
    zero = np.sum(likelihood[:,:,0:16], axis=2)
    one = np.sum(likelihood[:,:,16:32], axis=2)
    imagination = (one >= zero)*1

    print("Imagination of numbers in Bayesian classifier:\n")
    for i in range(NUM_CLASSES):
        print("{}:".format(i))
        for row in range (28):
            for col in range(28):
                print(imagination[i][row*28+col], end=' ')
            print(" ")
        print()
    return

def show_continuous_imagination(mean):
    imagination = (mean >= 128)*1

    print("Imagination of numbers in Bayesian classifier:\n")
    for i in range(NUM_CLASSES):
        print("{}:".format(i))
        for row in range (28):
            for col in range(28):
                print(imagination[i][row*28+col], end=' ')
            print(" ")
        print()
    return

if __name__ == '__main__':
    train_image, train_label = read_dataset('./data', train=True)
    test_image, test_label = read_dataset('./data', train=False)
    
    prior_prob = cal_prior_prob(train_label)
    # np.save("prior_prob.npy", prior_prob)
    # prior_prob = np.load("prior.npy")

    if args.mode == 0:
        # likelihood = cal_likelihood(train_image, train_label)
        # np.save("likelihood.npy", likelihood)
        likelihood = np.load("likelihood.npy")
        error_rate = predict_discrete_result(test_image, test_label, likelihood, prior_prob)
        show_discrete_imagination(likelihood)

    else:
        mean, var = cal_mean_var(train_image, train_label, prior_prob)
        # np.save("mean.npy", mean)
        # np.save('var.npy', var)
        # mean = np.load("mean.npy")
        # var = np.load("var.npy")
        error_rate = predict_continuous_result(test_image, test_label, mean, var, prior_prob)
        show_continuous_imagination(mean)

    print('Error rate:', error_rate)

