#Creator: Akshay Naik (aunaik@iu.edu)

import numpy as np

# Power Iteration
def power_iteration(X, iterations):
    y_km1 = np.random.rand(X.shape[0],1)
    for i in range(iterations):
        y_k = np.dot(X, y_km1)
        y_km1 = y_k/np.linalg.norm(y_k)
    return y_km1

# Calculating multiple Eigenvectors
def eigen_vectors(X, no_of_eigen_vectors, iterations):
    eigen_vector = np.zeros(X.shape[0])
    for i in range(no_of_eigen_vectors):
        v = power_iteration(X, iterations)
        inner_prod = np.dot(X, v)
        s = np.linalg.norm(inner_prod)
        u = inner_prod/s
        X = X - (s*np.dot(v, u.T))
        #print(str(i), " Eigen value: ", str(s))
        if i == 0:
            eigen_vector = v
        else:
            eigen_vector = np.append(eigen_vector, v, 1)
    return eigen_vector
