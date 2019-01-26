import cmath
import numpy as np
import librosa
import matplotlib.pyplot as plt


# Creating DFT matrix F
def DFT(x):
    F = np.exp(-2j * cmath.pi / N * np.dot(f.reshape(len(f), 1), n.reshape(len(n), 1).T))
    return F


# Creating inverse DFT matrix F_star
def I_DFT(x):
    IF = (1/N)*np.exp(2j * cmath.pi / N * np.dot(f.reshape(len(f), 1), n.reshape(len(n), 1).T))
    return IF


# Creating data matrix X
def create_X(x):
    for i in range(0,len(x),int(1024/2)):
        sig = x[i:(i + 1024)]
        l = len(sig)
        if l < 1024:
            break
        if i == 0:
            X= np.multiply(sig, han)
        else:
            X =np.c_[X, np.multiply(sig, han)]
    return X


#STFT
def stft(x):
    F = DFT(x)
    X = create_X(x)
    # STFT
    Y = np.dot(F,X)
    return Y[0:513,:]


# Cosine distance calculation
def cosine_dist(G, y_i, K):
    norm_curr_vec = np.linalg.norm(y_i)
    norm_data = np.linalg.norm(G, axis=1, keepdims=True)
    norm = np.multiply(norm_curr_vec, norm_data)
    temp_dist = 1 - (np.divide(np.dot(y_i, G.T).reshape(len(G), 1), norm))
    return temp_dist.flatten().argsort()[0:K]

# calling fn like print (cosine_dist(a.T,b.T[0], 1))


# Creating IBM of test signal
def create_IBM(G, B, Y, K):
    for i in range(Y.shape[1]):
        neighbours = cosine_dist(G.T,Y.T[i], K)
        if i==0:
            D = np.median(B[:,neighbours], axis = 1)
        else:
            D = np.c_[D, np.median(B[:,neighbours], axis = 1)]
    return D


# Signal reconstruction
def signal_reconstruction(X_estimated):
    X_estimated = X_estimated.T
    temp = np.zeros(len(x_t))
    for i in range(X_estimated.shape[0]):
        temp[(i*512):(i*512)+512] = temp[(i*512):(i*512)+512] + X_estimated[i,0:512].flatten()
        temp[((i*512)+512):((i*512)+1024)] = X_estimated[i,512:1024].flatten()
    return temp


# Variable initialization
N = 1024
K = 5
f, n =np.arange(N), np.arange(N)
han = np.hanning(1024).reshape(1024,1)


# For speaker signal trs.wav
x_s ,_ = librosa.load( 'data/trs.wav', sr=None)
x_s = x_s.reshape(len(x_s),1)


# STFT of signal trs.wav
S = np.abs(stft(x_s))


# For speaker signal trn.wav
x_n ,_ = librosa.load( 'data/trn.wav', sr=None)
x_n = x_n.reshape(len(x_n),1)

# STFT of signal trn.wav
N_sig = np.abs(stft(x_n))

x = (x_s+x_n)

# STFT of signal trn.wav
G = np.abs(stft(x))

B = np.zeros(G.shape)
B[S >= N_sig] = 1


# For noisy speech signal of the same speaker
x_t ,sr = librosa.load( 'data/x_nmf.wav', sr=None)
x_t = x_t.reshape(len(x_t),1)


# STFT of noisy signal
X = stft(x_t)
Y = np.abs(X)

D = create_IBM(G, B, Y, K)

# recovering speech source S_bar
S_bar_test = D * X
S_bar_test = np.r_[S_bar_test,np.conjugate(np.flip(S_bar_test[1:512],0))]


# Plotting the reconstructed spectrogram
# plt.figure(figsize=(4, 4))
# plt.imshow(np.abs(S_bar_test), cmap='jet', interpolation='none', aspect='auto')
# plt.axis('off')
# plt.show()


F_star = I_DFT(x_t)
S_bar_test = np.dot(F_star, S_bar_test)

# Recovering the time domain signal
recovered_signal = signal_reconstruction(S_bar_test.real)

# Plotting the time domain signals
# plt.plot(recovered_signal)
# plt.show()
#
# plt.plot(x_t)
# plt.show()

#librosa.output.write_wav('recovered_signal_Q4.wav', recovered_signal, sr)