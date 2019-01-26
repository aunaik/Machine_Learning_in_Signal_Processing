import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from copy import deepcopy


# Reading the data for june and december
june = loadmat('data/june.mat')['june'][:,0]
dec = loadmat('data/december.mat')['december'][:,0]

disparity = np.abs(dec - june).astype(np.int32)
#plt.hist(disparity,  bins=60)
#plt.savefig('hist.png')
#plt.show()

# Display output histogram
def histogram(data, clusters, c):
    label = ['Cluster 1', 'Cluster 2', 'Cluster Mean']
    color = ['b', 'g']
    for i in range(len(c)):
        points = [data[j] for j in range(len(data)) if clusters[j] == i]
        plt.hist(points, bins=55, alpha=0.5, facecolor=color[i], label=label[i])
        if i == 0:
            plt.axvline(x=c[i], color='r', label=label[2])
        else:
            plt.axvline(x=c[i], color='r')
    plt.legend(loc='upper right')
    plt.xlabel('Disparity')
    plt.ylabel('Number of stars')
    plt.title('K-means Clustering')
    #plt.savefig('K-means_hist.png')
    plt.show()


# Objective Function
def sum_of_square_error(data, clusters, c):
    SSE = 0
    for i in range(len(c)):
        points = [data[j] for j in range(len(data)) if clusters[j] == i]
        if len(points) != 0:
            SSE += np.sum(np.subtract(c[i], points)**2)
    return SSE


# Generating random centroid
def generate_centroids(k):
    #centroid = np.zeros(k)
    centroid = np.random.randint(1, 55, size=k)
    #centroid= np.array([41, 38])
    return centroid.astype(float)


# main function
def main(data, k):
    SSE = 0
    c = generate_centroids(k)
    c_old = np.zeros(c.shape)
    clusters = np.zeros(len(data))
    diff_in_clust = np.sum(np.abs(c - c_old))
    count = 0
    while diff_in_clust != 0:
        count += 1
        for i in range(len(data)):
            dist = np.abs(np.subtract(data[i], c))
            clusters[i] = np.argmin(dist)
        c_old = deepcopy(c)
        SSE = sum_of_square_error(data, clusters, c)
        for i in range(k):
            points = [data[j] for j in range(len(data)) if clusters[j] == i]
            if len(points) != 0:
                c[i] = np.mean(points)
        diff_in_clust = np.sum(np.abs(c - c_old))
    histogram(data, clusters, c)
    print("SSE: " + str(SSE))
    print("Centroids: ", c)
    print("Number of iterations: " + str(count))

main(disparity, 2)