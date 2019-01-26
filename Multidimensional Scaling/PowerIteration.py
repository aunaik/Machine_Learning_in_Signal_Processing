#Creator: Akshay Naik (aunaik@iu.edu)

import numpy as np

# Power Iteration
def power_iteration(X, iterations):
    y_km1 = np.ones((X.shape[0],1))#np.random.rand(X.shape[0],1)
    for i in range(iterations):
        y_k = np.dot(X, y_km1)
        s = np.linalg.norm(y_k)
        y_km1 = y_k/s
    return y_km1, s

# Calculating multiple Eigenvectors
def eigen_vectors(X, no_of_eigen_vectors, iterations):
    #eigen_vector = np.zeros(X.shape[0])
    #eigen_values = np.zeros()
    for i in range(no_of_eigen_vectors):
        v, s = power_iteration(X, iterations)
        #s = np.linalg.norm(inner_prod)
        u = np.dot(X.T, v)/s
        #v = v * s
        #print ("EV",v)
        #exit()
        X = X - (s*np.dot(v, u.T))
        #X = X - (np.dot(v, u.T))
        #print(str(i), " Eigen value: ", str(s))
        if i == 0:
            eigen_vector = v
            eigen_value = np.array(s)
        else:
            eigen_vector = np.append(eigen_vector, v, 1)
            eigen_value = np.append(eigen_value, s)
    return eigen_vector , eigen_value
