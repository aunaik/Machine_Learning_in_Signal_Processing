import numpy as np
import librosa
import matplotlib.pyplot as plt
import PowerIteration as pi

# Covariance matrix calculation
def my_cov(mat):
    #mat = mat.T
    mat = (mat.T - mat.mean(axis=1)).T
    return ((1 / float(mat.shape[1] - 1)) * np.dot(mat, mat.T)).squeeze()


# Save sound files
def save_signal(Y):
    for i in range(Y.shape[0]):
        librosa.output.write_wav('out_1_{}.wav'.format(i+1), Y[i], sr)


# Reading the data
for i in range(1,21):
    if i==1:
        X ,sr = librosa.load( 'data/x_ica_{}.wav'.format(i), sr=None)
        X = X.reshape(1,len(X))
        print("sampling frequency: ", sr)
    else:
        X = np.append(X,librosa.load('data/x_ica_{}.wav'.format(i), sr=None)[0].reshape(1,X.shape[1]),axis=0)


def ICA(X, learning_rate):
    # PCA
    W, A  = pi.eigen_vectors(my_cov(X), 4, 500)



    # Creating vector A of eigenvalues to a Diagonal matrix matrix
    A = np.power(A,-1/2)
    A = np.diag(A)


    # Whitening the data
    Z = np.dot(np.dot(A,W.T), X)


    # Initializing matrix W
    W = np.identity(Z.shape[0])


    Y = np.dot(W,Z)
    NI = Z.shape[1]*np.identity(Z.shape[0])

    conv = []

    # ICA
    for epoch in range(300):
        dW = np.dot((NI - np.dot(np.tanh(Y), np.power(Y, 3).T)), W)
        W = W + (learning_rate*dW)
        Y = np.dot(W,Z)
        #print (np.sum(np.abs(dW)))
        conv.append(np.sum(np.abs(dW)))

    plt.plot(conv)
    plt.title('Convergence Graph')
    plt.xlabel('Iterations')
    plt.ylabel('dW (Update on W)')
    plt.tight_layout()
    #plt.show()
    plt.savefig('convergence_graph.png')
    return Y

learning_rate = 0.000001

Y = ICA(X, learning_rate)
save_signal(Y)
