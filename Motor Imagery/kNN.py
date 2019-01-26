import cmath
import numpy as np
import scipy.io as sio
from scipy.stats.mstats import mode

eeg = sio.loadmat('data/eeg')
#print (eeg.keys())      # keys are 'x_te', 'x_train', 'y_te', 'y_train'

X_train = eeg['x_train']
Y_train = eeg['y_train']
X_test = eeg['x_te']
Y_test = eeg['y_te']

N = 64
f, n =np.arange(N), np.arange(N)
bm = np.blackman(N)


# Creating DFT matrix F
def DFT(x):
    F = np.exp(-2j * cmath.pi / N * np.dot(f.reshape(len(f), 1), n.reshape(len(n), 1).T))
    return F


# Creating data matrix X
def create_X(x):
    for i in range(0,len(x),48):
        sig = x[i:(i + 64)]
        l = len(sig)
        if l < 64:
            break
        if i == 0:
            X= np.multiply(sig, bm)
        else:
            X =np.c_[X, np.multiply(sig, bm)]
    return X


# STFT
def stft(x):
    F = DFT(x)
    X = create_X(x)
    Y = np.dot(F,X)[3:8].reshape(-1)
    return np.abs(Y)


# Creating data matrix Z
def create_Z(X):
    # Loop over 112
    for i in range(X.shape[2]):
        if i == 0:
            Z = np.hstack((stft(X[:,0,i]),stft(X[:,1,i]),stft(X[:,2,i]))).reshape(-1,1)
        else:
            Z = np.c_[Z,np.hstack((stft(X[:,0,i]),stft(X[:,1,i]),stft(X[:,2,i]))).reshape(-1,1)]
    return Z

# Creating random projection matrix
def rand_proj_mat(L, M):
    #np.random.seed(28)
    A = np.random.uniform(-1,1,size=(L, M))
    A = A/A.sum(axis=1)[:,np.newaxis]
    return A


# Reducing dimension by doing random projection and returning binary features of dimension L
def dim_reduction(X, L=2, M=225):
    return np.sign(np.dot(rand_proj_mat(L, M), create_Z(X)))


# Calculating hamming distance between a vector and a matrix
def hamming(X_train, i, K = 5):
    #print ((np.sum(np.abs(i.reshape(-1, 1) - X_train_reduced_dim), axis=0) * 0.5).argsort())
    return ((np.sum(np.abs(i.reshape(-1, 1) - X_train_reduced_dim), axis=0) * 0.5).argsort())[:K]

# Set value of M
M = 225

K_list =[2, 5, 10, 15, 20]
L_list = [2, 4, 6, 8,10, 15, 20]
for l in L_list:
    # Set value of L
    L = l
    for k in K_list:
        # Set value of K
        K = k
        Accuracy = []
        np.random.seed(1)
        for i in range(30):
            # Performing dimensionality reduction on both train and test data
            X_train_reduced_dim = dim_reduction(X_train, L, M)
            X_test_reduced_dim = dim_reduction(X_test, L, M)
            X_train_reduced_dim[X_train_reduced_dim == -1] = 0
            X_test_reduced_dim[X_test_reduced_dim == -1] = 0


            y_pred = []
            # Predicting class of test data using K-NN
            for i in range(Y_test.shape[0]):
                neighbours_indexs = hamming(X_train_reduced_dim, X_test_reduced_dim[:,i], K)
                neighbours_class = list(Y_train[neighbours_indexs,0])
                y_pred.append(max(neighbours_class, key=neighbours_class.count))

            Accuracy.append(sum(Y_test.flatten() == np.array(y_pred))*100/28)


        #print (Accuracy)
        print ("Accuracies with K=", K," and L=", L, ": ", max(Accuracy))
