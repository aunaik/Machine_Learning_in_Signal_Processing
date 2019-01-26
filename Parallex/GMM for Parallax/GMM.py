import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Reading the data for june and december
june = loadmat('data/june.mat')['june'][:,0]
dec = loadmat('data/december.mat')['december'][:,0]

disparity = np.abs(dec - june).astype(np.int32)


# Random initialization of the priors as well as distribution parameters
def init_em(k):
    means, std = np.random.randint(5, 50, size=k), np.random.randint(4, 10, size=k)
    priors = np.random.dirichlet((1,)*k, 1).flatten()
    #means = np.array([41, 38])
    #std = np.array([4, 6])
    #priors = np.array([0.5429364, 0.4570636])
    #print (means, std, priors)
    return means, std, priors


# Display output histogram
def histogram_em(data, w, means):
    label = ['Cluster 1', 'Cluster 2', 'Cluster Mean']
    color = ['b', 'g']
    clusters = np.argmax(w, axis=1)
    x = []
    for i in range(len(means)):
        points = [data[j] for j in range(len(data)) if clusters[j] == i]
        x =np.r_[x,points]
        plt.hist(points,  bins=55, alpha=0.5, facecolor=color[i], label=label[i])
        if i == 0:
            plt.axvline(x=means[i], color='r', label=label[2])
        else:
            plt.axvline(x=means[i], color='r')
    plt.legend(loc='upper right')
    plt.xlabel('Disparity')
    plt.ylabel('Number of stars')
    plt.title('EM Clustering')
    #plt.savefig('EM_hist.png')
    plt.show()


# Objective Function
def sum_of_square_error(data, w, means):
    SSE = 0
    clusters = np.argmax(w, axis=1)
    for i in range(len(means)):
        points = [data[j] for j in range(len(data)) if clusters[j] == i]
        if len(points) != 0:
            SSE += np.sum(np.subtract(means[i], points)**2)
    return SSE


# Likelihood Probability function
def likelihood_em(i, mean, std):
    return (1/std)*(np.exp(- ((i - mean)**2)/(2*(std**2))))

# Log likelihood function for convergence evaluation
def eval_em(w):
    return (np.sum(np.log(np.sum(w, axis=1))))


def main_em(data, k):
    # Initializing the parameters
    means, std, priors = init_em(k)
    ll_old, ll = 0, 100
    count = 0
    while abs(ll_old - ll) >= 0.001:
        count += 1

        # Assigning new log likelihood to old log likelihood
        ll_old = ll

        # E-step
        w = np.c_[likelihood_em(data, means[0], std[0]).reshape(len(data), 1), likelihood_em(data, means[1], std[1]).reshape(len(data), 1)]
        w = w*priors

        w_norm = np.divide(w, np.sum(w, axis=1).reshape(len(data),1))

        # M-step
        priors = (1/len(data)) * np.sum(w_norm, axis= 0)
        means = np.sum(w_norm*data.reshape(len(data),1), axis=0)/np.sum(w_norm, axis=0)
        std = np.sqrt(np.sum(w_norm * ((data.reshape(len(data),1)-means)**2), axis= 0)/np.sum(w_norm, axis=0))

        ll = eval_em(w)
    histogram_em(data, w_norm, means)
    print ("Cluster Means: ", means)
    print ("Cluster Standard Deviation: ", std)
    print ("Number of iterations: ", count)

main_em(disparity, 2)