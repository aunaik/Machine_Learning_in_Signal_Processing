import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import PowerIteration as pi
from mpl_toolkits.mplot3d import Axes3D

# Creating distance matrix
def create_mat(X):
    for i in range(X.shape[0]):
        if i==0:
            Z = np.exp(-(np.linalg.norm(X[i] - X, axis = 1)**2/(2*sigma**2))).reshape(1,X.shape[0])
        else:
            Z = np.r_[Z,  np.exp(-(np.linalg.norm(X[i] - X, axis=1) ** 2 / (2 * sigma ** 2))).reshape(1,X.shape[0])]
    return Z

# Creating 3d plot
def threeD_plot(Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z[0,:51], Z[1,:51], Z[2,:51], c='r', marker='o')
    ax.scatter(Z[0, 51:152], Z[1, 51:152], Z[2, 51:152], c='b', marker='^')
    ax.set_xlabel('Axis 1')
    ax.set_ylabel('Axis 2')
    ax.set_zlabel('Axis 3')
    plt.show()


# Activation function for each perceptron in neural network
def sigmoidvalue(mat):
    #mat = np.clip(mat, -500, 500)
    return 1.0 / (1 + np.exp(-mat))


# Training a perceptron that does linear classification
def perceptron(Z):
    # Initializing random weights
    np.random.seed(1)
    w = np.random.uniform(-0.5, 0.5, size=(1, 3))
    b = np.random.uniform(-0.5, 0.5, size=1)

    # Initializing the ground-truth output
    t = np.c_[np.zeros((1,51)), np.ones((1,101))]

    # Initializing the learning rate and number of data points
    alpha = 0.28
    n = Z.shape[1]
    L = []
    # TRAINING
    for epoch in range(10000):
        # Forward propogation
        y = sigmoidvalue(np.dot(w, Z) + b)

        # Loss calculation
        Loss = np.sum(0.5 * (y-t)**2)
        L.append(Loss)
        print (Loss)

        # Back propogation
        delta = y * (1-y) * (y-t)
        dw = np.dot(delta, Z.T)

        # Updating weights
        w = w - (alpha/n) * dw
        b = b - (alpha/n) * np.sum(delta)

    # TESTING
    y = sigmoidvalue(np.dot(w, Z) + b)
    #print (np.round(y,0))
    #print (t)

    plt.plot(L)
    plt.title('Error VS Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

    print ('Accuracy of classification is : {}%'.format((np.sum(np.abs(np.round(y,0) == t))*100)/y.shape[1]))



def main():
    # Loading the 2-dimensional data
    X = loadmat('data/concentric.mat')['X']

    # Plotting the data points
    plt.scatter(X[0, :], X[1, :])
    plt.show()

    # Call to create distance matrix
    Y = create_mat(X.T)
    print (Y.shape)

    W, A = pi.eigen_vectors(Y, 3, 500)

    print (W.shape)

    # Kernel PCA
    Z = np.dot(W.T, Y)
    print (Z.shape)

    # Creating 3d plot
    threeD_plot(Z)

    # Calling perceptron training function
    perceptron(Z)

sigma = 0.1
main()