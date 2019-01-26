import cmath
import numpy as np
import librosa
import matplotlib.pyplot as plt


# Initialize matrix W and H
def init_W_H(F, B, T, init_W = True):
    W = 0
    if init_W == True:
        W = np.random.uniform(low=0.1, high=4, size=(F,B))
    H = np.random.uniform(low=0.1, high=4, size=(B, T))
    return W, H


# Error function
def err_fn(X , X_bar):
    return np.sum(X*np.log(X/X_bar) - X + X_bar)


# NMF
def NMF(X, W, H, update_W = True):
    ones = np.ones((X.shape))
    prev_error = 99999
    i = 0
    if update_W == True:
        while(True):
            i += 1
            X_bar = np.dot(W, H)
            W = W * (np.dot(X/(np.dot(W, H)+np.finfo(float).eps), H.T))/(np.dot(ones, H.T)+np.finfo(float).eps)
            H = H * (np.dot(W.T, (X/(np.dot(W, H)+np.finfo(float).eps))))/(np.dot(W.T, ones)+np.finfo(float).eps)
            err = err_fn(X, X_bar)
            #print (err, abs(prev_error - err), i)
            if abs(prev_error - err) < 1:
                break
            prev_error = err
    else:
        while (True):
            i+=1
            X_bar = np.dot(W, H)
            X_WH = X / (np.dot(W, H)+np.finfo(float).eps)
            H = H * np.dot(W.T, X_WH) / (np.dot(W.T, ones)+np.finfo(float).eps)
            err = err_fn(X, X_bar)
            #print(err, abs(prev_error - err), i)
            if abs(prev_error - err) < 1:
                break
            prev_error = err
    return W, H


# Creating inverse DFT matrix F_star
def I_DFT(x):
    IF = (1/N)*np.exp(2j * cmath.pi / N * np.dot(f.reshape(len(f), 1), n.reshape(len(n), 1).T))
    return IF

# Creating DFT matrix F
def DFT(x):
    F = np.exp(-2j * cmath.pi / N * np.dot(f.reshape(len(f), 1), n.reshape(len(n), 1).T))
    return F

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


# Signal reconstruction
def signal_reconstruction(X_estimated):
    X_estimated = X_estimated.T
    temp = np.zeros(len(x_t))
    for i in range(X_estimated.shape[0]):
        temp[(i*512):(i*512)+512] = temp[(i*512):(i*512)+512] + X_estimated[i,0:512].flatten()
        temp[((i*512)+512):((i*512)+1024)] = X_estimated[i,512:1024].flatten()
    return temp


N = 1024
f, n =np.arange(N), np.arange(N)
han = np.hanning(1024).reshape(1024,1)

print (I_DFT(f))

# For speaker signal trs.wav
x_s ,_ = librosa.load( 'data/trs.wav', sr=None)
x_s = x_s.reshape(len(x_s),1)

# STFT of signal trs.wav
S = np.abs(stft(x_s))

# Initializing matrix W and H
W_S, H_S = init_W_H(S.shape[0], 30, S.shape[1])

# Non-negative Matrix Factorization
W_S, H_S = NMF(S, W_S, H_S)


# For speaker signal trn.wav
x_n ,_ = librosa.load( 'data/trn.wav', sr=None)
x_n = x_n.reshape(len(x_n),1)

# STFT of signal trn.wav
N_sig = np.abs(stft(x_n))

# Initializing matrix W and H
W_N, H_N = init_W_H(N_sig.shape[0], 30, N_sig.shape[1])

# Non-negative Matrix Factorization
W_N, H_N = NMF(N_sig, W_N, H_N)


# For noisy speech signal of the same speaker
x_t ,sr = librosa.load( 'data/x_nmf.wav', sr=None)
x_t = x_t.reshape(len(x_t),1)
print ("sampling frequency: ", sr)

# STFT of noisy signal
X = stft(x_t)
Y = np.abs(X)

# Initializing matrix W and H
_, H = init_W_H(Y.shape[0], 60, Y.shape[1], False)

W = np.c_[W_S, W_N]

# Non-negative Matrix Factorization
_, H = NMF(Y, W, H, False)

# Calculating magnitude masking matrix
M_bar = np.dot(W_S, H[0:30,:])/np.dot(W, H)

# recovering speech source S_bar
S_bar = M_bar * X
S_bar = np.r_[S_bar,np.conjugate(np.flip(S_bar[1:512],0))]

# Plotting the reconstructed spectrogram
# plt.figure(figsize=(4, 4))
# plt.imshow(np.abs(S_bar), cmap='jet', interpolation='none', aspect='auto')
# plt.axis('off')
# plt.show()

F_star = I_DFT(x_t)
S_bar = np.dot(F_star, S_bar)

# Recovering the time domain signal
recovered_signal = signal_reconstruction(S_bar.real)

# Plotting the time domain signals
# plt.plot(recovered_signal)
# plt.show()

# plt.plot(x_t)
# plt.show()

librosa.output.write_wav('recovered_signal.wav', recovered_signal, sr)


# Checking why we want 513 frequency bands instead of 512 during removing the lower frequency band out of 1024
# X = np.random.randint(1,50,size=16)
# N=16
# f, n =np.arange(16),np.arange(16)
# F = DFT(X)
# print (np.dot(F,X))