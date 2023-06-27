import numpy as np
import torch


# import stft_64_pad_0 as stft
import stft_64 as stft
from audio_processing import griffin_lim

from scipy.io.wavfile import read
import time

def magnitude_to_L(magnitude, forward_basis):
    # magnitude:      [1, 2H+1, T]
    # forward_basis:  [2H+1, 1, 4H]
    
    # output L shape: [2H+1,T]
    H = (magnitude.shape[1]-1)//2
    basis = np.zeros((2*H+1,2*H+1), dtype= np.float64)
    basis[:, :] = forward_basis[:,0,:2*H+1]
    basis_inv = np.linalg.pinv(basis)
    L = np.matmul(basis_inv, magnitude[0,:,:]**2)
    L[1:2*H] = L[1:2*H]/2
    return L
    