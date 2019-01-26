import numpy as np
import librosa
import matplotlib.pyplot as plt
import PowerIteration as pi


x ,sr = librosa.load( 'data/s.wav', sr=None)
x = x.reshape(len(x),1)
print ("sampling frequency: ", sr)
print (x.shape)


# Randomly selecting 8 consecutive samples
def sampling(x, n):
    np.random.seed(100)
    print (n)
    s = np.random.randint(0, len(x) - 9, n)
    print(np.min(s), np.max(s))
    for i in range(len(s)):
        temp_vec = x[s[i]:(s[i]+8)]
        temp_vec = np.power(temp_vec,2)
        # temp_vec = np.abs(temp_vec)
        if i == 0:
            X = temp_vec.reshape(len(temp_vec),1)
        else:
            X = np.c_[X,temp_vec.reshape(len(temp_vec),1)]
    return X


# Covariance matrix calculation
def my_cov(mat):
    mat = (mat.T - mat.mean(axis=1)).T
    return ((1 / float(mat.shape[1])) * np.dot(mat, mat.T)).squeeze()


# PCA
def PCA(X, no_of_eigen_vectors, iterations):
    return pi.eigen_vectors(my_cov(X), no_of_eigen_vectors, iterations)


# For 10 samples
X = sampling(x,10)
V = PCA(X, 8, 600)
plt.figure(figsize = (4,4))
plt.imshow(V.T, interpolation='none', aspect ='auto')
plt.axis('off')
#plt.savefig("10-samples.png")
plt.show()

# For 100 samples
X = sampling(x,100)
V = PCA(X, 8, 600)
plt.figure(figsize = (4,4))
plt.imshow(V.T, interpolation='none', aspect ='auto')
plt.axis('off')
#plt.savefig("100-samples.png")
plt.show()

# For 1000 samples
X = sampling(x,1000)
V = PCA(X, 8, 600)
plt.figure(figsize = (4,4))
plt.imshow(V.T, interpolation='none', aspect ='auto')
plt.axis('off')
#plt.savefig("1000-samples.png")
plt.show()

# Plotting for 10000 samples and 100000 samples
# # For 10000 samples
# X = sampling(x,10000)
# V = PCA(X, 8, 600)
# plt.figure(figsize = (4,4))
# plt.imshow(V.T, interpolation='none', aspect ='auto')
# plt.axis('off')
# #plt.savefig("10000-samples.png")
# plt.show()
#
# # For 100000 samples
# X = sampling(x,100000)
# V = PCA(X, 8, 600)
# plt.figure(figsize = (4,4))
# plt.imshow(V.T, interpolation='none', aspect ='auto')
# plt.axis('off')
# #plt.savefig("100000-samples.png")
# plt.show()
