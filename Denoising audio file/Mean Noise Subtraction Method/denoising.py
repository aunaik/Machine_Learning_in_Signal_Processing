import cmath
import numpy as np
import librosa
import matplotlib.pyplot as plt

x ,sr = librosa.load( 'data/x.wav', sr=None)
x = x.reshape(len(x),1)
print ("sampling frequency: ", sr)


N = 1024
f, n =np.arange(N), np.arange(N)
han = np.hanning(1024).reshape(1024,1)

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


# Signal reconstruction
def signal_reconstruction(X_estimated):
    X_estimated = X_estimated.T
    temp = np.zeros(len(x))
    print (X_estimated.shape[0])
    for i in range(X_estimated.shape[0]):
        temp[(i*512):(i*512)+512] = temp[(i*512):(i*512)+512] + X_estimated[i,0:512].flatten()
        temp[((i*512)+512):((i*512)+1024)] = X_estimated[i,512:1024].flatten()
    return temp


def stft(x):
    F = DFT(x)
    X = create_X(x)

    # STFT
    Y = np.dot(F,X)

    #Plotting the signal in time-frequency domain
    plt.figure(figsize=(4, 4))
    plt.imshow(np.power(np.abs(Y),0.5), cmap='jet', interpolation='none', aspect='auto')
    plt.axis('off')
    #plt.savefig('Spectorgram.png')
    plt.show()

    # Noise model
    Y_noise = np.c_[Y[:, 0:40], Y[:, 150:186]] #np.c_[Y[:, 0:12], Y[:, 170:186]]
    M = np.mean(np.mean(np.abs(Y_noise), axis = 1).reshape(1024,1))

    # Subtracting the noise vector to get residual signal
    residual_magnitude = np.abs(Y) - M
    residual_magnitude[residual_magnitude < 0] = 0

    # Phase
    theta = Y/np.abs(Y)

    # Estimated clean spectra
    est_clean_spectra = np.multiply(theta, residual_magnitude)

    # plt.figure(figsize=(4, 4))
    # plt.imshow(np.power(np.abs(residual_magnitude), 0.5), cmap='jet', interpolation='none', aspect='auto')
    # plt.axis('off')
    # plt.show()

    # Inverse-STFT
    F_star = I_DFT(x)

    print (F_star.shape, residual_magnitude.shape)
    print (X.shape)


    X_estimated = np.dot(F_star, est_clean_spectra)
    X_estimated = X_estimated.real
    print (X_estimated.shape)
    op = signal_reconstruction(X_estimated)
    return op



op = stft(x)

librosa.output.write_wav('output.wav', op, sr)