from scipy.io import loadmat
import matplotlib.pyplot as plt
import PowerIteration as pi

L = loadmat('data/MDS_pdist.mat')['L']

# MDS
L = L - L.mean(axis = 1)
L = L - L.mean(axis = 0)

W, A  = pi.eigen_vectors(L, 2, 500)

plt.scatter(W[:,0], W[:,1])
plt.show()