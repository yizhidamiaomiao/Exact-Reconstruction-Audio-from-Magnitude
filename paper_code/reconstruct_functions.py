import IPython.display as ipd

import numpy as np
import torch


from scipy.io.wavfile import read
import time
from tqdm import tqdm



def compare(a,b):
    return torch.mean(torch.abs(a-b)), torch.mean((a-b)*(a-b))

def compare_L1(ori,gen):
    return torch.mean(torch.abs(ori-gen)/torch.abs(ori))


def compare_L2(a,b):
    return torch.sum(torch.abs(a-b)), torch.sum((a-b)*(a-b))


def magnitude_to_L(magnitude, forward_basis):
    # magnitude:      [1, 2H+1, T]
    # forward_basis:  [2H+1, 1, 4H]
    
    # output L shape: [2H+1,T]
    H = (magnitude.shape[1]-1)//2
    basis = np.zeros((2*H+1,2*H+1), dtype= np.float64)
    basis[:, :] = forward_basis[:,0,:2*H+1]
    basis_inv = np.linalg.pinv(basis)
    L = np.matmul(basis_inv, magnitude[0,:,:]**2)
    L[1:2*H+1] = L[1:2*H+1]/2
    return L

def reconstruct_from_S_with_H_2(magnitude, stft_fn, hop_length=2, frame_number = None):
    # input shape: [1, channels, :]
    channels = magnitude.shape[1]
    win_length = (channels-1)*2
    
    recon = np.zeros((1, win_length - hop_length + hop_length * magnitude.shape[-1])).astype(np.float64)
    
    #0-2 window
    recon[0,:win_length+2*hop_length],_ = reconstruct_from_S_with_H_2_block(magnitude[:,:,0:3], stft_fn, hop_length)
    
    if frame_number == None:
        frame_number = magnitude.shape[-1]-2
    #i-(i+2) window
    for i in tqdm(range(1, frame_number)):
        recon_current,_ = reconstruct_from_S_with_H_2_block(magnitude[:,:,i:i+3], stft_fn, hop_length)
        modified_sign =1
        if recon_current[0,0]>0 and recon[0,hop_length*i]<0:
            modified_sign = -1
        elif recon_current[0,0]<0 and recon[0,hop_length*i]>0:
            modified_sign = -1
        recon[0,win_length - hop_length + hop_length*(i+2) :win_length + hop_length*(i+2)] = \
                  modified_sign * recon_current[0, - hop_length :]
    return recon
    

def reconstruct_from_S_with_H_2_block(magnitude, stft_fn, hop_length=2):
    #input shape: [1, channels, 3]
    
    channels = magnitude.shape[1]
    win_length = (channels-1)*2
    L = magnitude_to_L(magnitude, stft_fn.forward_basis[:2*hop_length+1,:,:])
    L = L.numpy()
    # print('R shape', L.shape)
    minimum_error = 1e10
    minimum_ans = np.zeros((1, win_length+2*hop_length)).astype(np.float64)

    for i in range(2**6):
        sgn10 = (i%(2**1))//(2**0)
        # sgn10 = 1 
        T10 = (sgn10 * 2 - 1) * np.sqrt(L[0,0] + 2*L[1,0] + 2*L[2,0] + 2*L[3,0] + 2*L[4,0])
        sgn20 = (i%(2**2))//(2**1)
        # sgn20 = 0
        T20 = (sgn20 * 2 - 1) * np.sqrt(L[0,0] - 2*L[1,0] + 2*L[2,0] - 2*L[3,0] + 2*L[4,0])
        S10 = (T10+T20)/2
        S20 = (T10-T20)/2
    
        sgn11 = (i%(2**3))//(2**2)
        # sgn11 = 1 
        T11 = (sgn11 * 2 - 1) * np.sqrt(L[0,1] + 2*L[1,1] + 2*L[2,1] + 2*L[3,1] + 2*L[4,1])
        sgn21 = (i%(2**4))//(2**3)
        # sgn21 = 0
        T21 = (sgn21 * 2 - 1) * np.sqrt(L[0,1] - 2*L[1,1] + 2*L[2,1] - 2*L[3,1] + 2*L[4,1])
        S11 = (T11+T21)/2
        S21 = (T11-T21)/2
    
        sgn12 = (i%(2**5))//(2**4)
        # sgn12 = 1 
        T12 = (sgn12 * 2 - 1) * np.sqrt(L[0,2] + 2*L[1,2] + 2*L[2,2] + 2*L[3,2] + 2*L[4,2])
        sgn22 = (i%(2**6))//(2**5)
        # sgn22 = 0
        T22 = (sgn22 * 2 - 1) * np.sqrt(L[0,2] - 2*L[1,2] + 2*L[2,2] - 2*L[3,2] + 2*L[4,2])
        S12 = (T12+T22)/2
        S22 = (T12-T22)/2
    
        x8_0 = S11 - S10
        x9_1 = S21 - S20
        x10_2 = S12 - S11
        x11_3 = S22 - S21
    
        A_matrix = np.zeros((channels*2, win_length), dtype=np.double)
        b = np.zeros((channels*2,1), dtype=np.double)
    
    
        ### equations for 0-4
        A_matrix[0,0] = 2*x8_0
        A_matrix[0,1] = 2*x9_1
        b[0,0] = L[0,1] - L[0,0] - x8_0**2 - x9_1**2
    
    
        A_matrix[1,1], A_matrix[1,7] = x8_0, x8_0
        A_matrix[1,0], A_matrix[1,2] = x9_1, x9_1
        b[1,0] = L[1,1] - L[1,0]-x9_1*x8_0
    
        A_matrix[2,2], A_matrix[2,6] = x8_0, x8_0
        A_matrix[2,3], A_matrix[2,7] = x9_1, x9_1
        b[2,0] = L[2,1] - L[2,0]
    
        A_matrix[3,3], A_matrix[3,5] = x8_0, x8_0
        A_matrix[3,4], A_matrix[3,6] = x9_1, x9_1
        b[3,0] = L[3,1] - L[3,0]
    
        A_matrix[4,4]= x8_0
        A_matrix[4,5]= x9_1
        b[4,0] = L[4,1] - L[4,0]
    
        ### equations for 5-9
        A_matrix[5,2] = 2*x10_2
        A_matrix[5,3] = 2*x11_3
        b[5,0] = L[0,2] - L[0,1] - x10_2**2 - x11_3**2
    
    
        A_matrix[6,1], A_matrix[6,3] = x10_2, x10_2
        A_matrix[6,2], A_matrix[6,4] = x11_3, x11_3
        b[6,0] = L[1,2] - L[1,1] - x10_2*x9_1 - x11_3*x10_2
    
        A_matrix[7,0], A_matrix[7,4] = x10_2, x10_2
        A_matrix[7,1], A_matrix[7,5] = x11_3, x11_3
        b[7,0] = L[2,2] - L[2,1] - x10_2*x8_0 - x11_3*x9_1
    
        A_matrix[8,7], A_matrix[8,5] = x10_2, x10_2
        A_matrix[8,0], A_matrix[8,6] = x11_3, x11_3
        b[8,0] = L[3,2] - L[3,1] - x11_3*x8_0
    
        A_matrix[9,6]= x10_2
        A_matrix[9,7]= x11_3
        b[9,0] = L[4,2] - L[4,1]
        
        ######### Using qr factorization
        Q,R = np.linalg.qr(A_matrix) # qr decomposition of A
        Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
        try:
            ans = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
        except:
            print('One sign meet singular matrix, please check original signal for detecting outliers')
            continue
    
    
        ######### Wrong using A^TA=A^Tb
#         temp = A_matrix.T @ A_matrix
#         ans = (np.linalg.inv(temp)) @ (A_matrix.T @ b)
        
#         if np.linalg.matrix_rank(temp)<win_length:
#             print('not full rank, may be error')
        
#         b_hat = A_matrix @ ans
    
        test_L   = np.zeros((channels, 3)).astype(np.float64)
        ans_full = np.zeros((1, win_length + 2*hop_length)).astype(np.float64)
        ans_full[0, :win_length] = ans[:win_length, 0]
        ans_full[0, win_length]   = ans[0,0]+x8_0
        ans_full[0, win_length+1] = ans[1,0]+x9_1
        ans_full[0, win_length+2] = ans[2,0]+x10_2
        ans_full[0, win_length+3] = ans[3,0]+x11_3
        
        for tt in range(3):
            for i in range(2*hop_length+1):
                test_L[i,tt] = np.sum(ans_full[0, tt*hop_length:tt*hop_length+win_length] * \
                                       np.roll(ans_full[0, tt*hop_length:tt*hop_length+win_length],i)
                                      )
        test_L[-1, :] = test_L[-1, :]/2
    
    
        if np.sum(np.abs(test_L - L))<minimum_error:
            minimum_error = np.sum(np.abs(test_L - L))
            minimum_ans[0,:] = ans_full[0,:]
    return minimum_ans, minimum_error





def reconstruct_from_S_with_H_1(magnitude, stft_fn, hop_length=1, frame_number = None):
    # input shape: [1, channels, :]
    channels = magnitude.shape[1]
    win_length = (channels-1)*2
    
    recon = np.zeros((1, win_length - hop_length + hop_length * magnitude.shape[-1])).astype(np.float64)
    
    #0-2 window
    recon[0,:win_length+2*hop_length],_ = reconstruct_from_S_with_H_1_block(magnitude[:,:,0:3], stft_fn, hop_length)
    
    if frame_number == None:
        frame_number = magnitude.shape[-1]-2
    #i-(i+2) window
    for i in tqdm(range(1, frame_number)):
        recon_current,_ = reconstruct_from_S_with_H_1_block(magnitude[:,:,i:i+3], stft_fn, hop_length)
        modified_sign =1
        if recon_current[0,0]>0 and recon[0,hop_length*i]<0:
            modified_sign = -1
        elif recon_current[0,0]<0 and recon[0,hop_length*i]>0:
            modified_sign = -1
        recon[0,win_length - hop_length + hop_length*(i+2) :win_length + hop_length*(i+2)] = \
                  modified_sign * recon_current[0, - hop_length :]
    return recon
    

def reconstruct_from_S_with_H_1_block(magnitude, stft_fn, hop_length=1):
    #input shape: [1, channels, 3]
    
    channels = magnitude.shape[1]
    win_length = (channels-1)*2
    L = magnitude_to_L(magnitude, stft_fn.forward_basis[:2*hop_length+1,:,:])
    L = L.numpy()
    # print('R shape', L.shape)
    minimum_error = 1e10
    minimum_ans = np.zeros((1, win_length+2*hop_length)).astype(np.float64)

    for i in range(2**3):
        sgn10 = (i%(2**1))//(2**0)
        S0 = (sgn10 * 2 - 1) * np.sqrt(L[0,0] + 2*L[1,0] + 2*L[2,0])
    
        sgn11 = (i%(2**2))//(2**1)
        S1 = (sgn11 * 2 - 1) * np.sqrt(L[0,1] + 2*L[1,1] + 2*L[2,1])
    
        sgn12 = (i%(2**5))//(2**4)
        # sgn12 = 1 
        S2 = (sgn12 * 2 - 1) * np.sqrt(L[0,2] + 2*L[1,2] + 2*L[2,2])
    
        x4_0 = S1 - S0
        x5_1 = S2 - S1
    
        A_matrix = np.zeros((channels*2, win_length), dtype=np.double)
        b = np.zeros((channels*2,1), dtype=np.double)
    
    
        ### equations for 0-2
        A_matrix[0,0] = 2*x4_0
        b[0,0] = L[0,1] - L[0,0] - x4_0**2
    
        A_matrix[1,1], A_matrix[1,3] = x4_0, x4_0
        b[1,0] = L[1,1] - L[1,0]
    
        A_matrix[2,2]= x4_0
        b[2,0] = L[2,1] - L[2,0]
    
        ### equations for 3-5
        A_matrix[3,1] = 2*x5_1
        b[3,0] = L[0,2] - L[0,1] - x5_1**2
    
        A_matrix[4,0], A_matrix[4,2] = x5_1, x5_1
        b[4,0] = L[1,2] - L[1,1] - x4_0*x5_1
    
        A_matrix[5,3]= x5_1
        b[5,0] = L[2,2] - L[2,1]
        
        ######### Using qr factorization
        Q,R = np.linalg.qr(A_matrix) # qr decomposition of A
        Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
        try:
            ans = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
        except:
            print('One sign meet singular matrix, please check original signal for detecting outliers')
            continue
    
    
        ######### Wrong using A^TA=A^Tb
#         temp = A_matrix.T @ A_matrix
#         ans = (np.linalg.inv(temp)) @ (A_matrix.T @ b)
        
#         if np.linalg.matrix_rank(temp)<win_length:
#             print('not full rank, may be error')
        
#         b_hat = A_matrix @ ans
    
        test_L   = np.zeros((channels, 3)).astype(np.float64)
        ans_full = np.zeros((1, win_length + 2*hop_length)).astype(np.float64)
        ans_full[0, :win_length] = ans[:win_length, 0]
        ans_full[0, win_length]   = ans[0,0]+x4_0
        ans_full[0, win_length+1] = ans[1,0]+x5_1
        
        for tt in range(3):
            for i in range(2*hop_length+1):
                test_L[i,tt] = np.sum(ans_full[0, tt*hop_length:tt*hop_length+win_length] * \
                                       np.roll(ans_full[0, tt*hop_length:tt*hop_length+win_length],i)
                                      )
        test_L[-1, :] = test_L[-1, :]/2
    
    
        if np.sum(np.abs(test_L - L))<minimum_error:
            minimum_error = np.sum(np.abs(test_L - L))
            minimum_ans[0,:] = ans_full[0,:]
    return minimum_ans, minimum_error


def Yw_to_L(magnitude, T,N,H):
    # input magnitude shape[(T+N-1)/H, T/2+1]
    # output L shape: [(T+N-1)/H, T/2+1]
    T = (magnitude.shape[1]-1)*2
    magnitude = magnitude.T
    basis = np.zeros((T//2+1,T//2+1), dtype= np.float64)
    for n in range(T//2+1):
        for c in range(T//2+1):
            basis[n,c] = np.cos(-2*np.pi*n*c/T)
    basis_inv = np.linalg.pinv(basis)
    L = np.matmul(basis_inv, magnitude[:,:]**2) #L shape[T/2+1, (T+N-1)/H]
    L = L.T # L shape[(T+N-1)/H, T/2+1]
    L[:,1:T//2+1] = L[:, 1:T//2+1]/2
    return L

def reconstruct_from_Yw_with_H_1(magnitude, window, hop_length=1):
    
    # input window shape: [N]
    # input magnitude shape[(T+N-1)/H, T/2+1]
    # output shape: [1, T]
    T = (magnitude.shape[-1]-1)*2
    H = hop_length
    N = window.shape[0]
    L = Yw_to_L(magnitude, T,N,H)# L shape[(T+N-1)/H, T/2+1]
    # print(L)

    
    ans_full = np.zeros((1, T)).astype(np.float64)
    x0 = torch.sqrt(L[0,0])/window[0]
    x1 = L[1,1]/(x0* window[1]*window[0])
    ans_full[0,0] = x0
    ans_full[0,1] = x1
    
    for m in tqdm(range(2, L.shape[0]-3)):
        A_matrix = np.zeros((H+1, H), dtype=np.double)
        b = np.zeros((H+1,1), dtype=np.double)
        for n in range(H, 2*H):
            b[n-H,0] = L[m,n]
            for t in range(T):
                i=t
                j=(t+n)%T
                if m*H-i<0 or m*H-j<0 or m*H-i>=N or m*H-j>=N:
                    continue
                if i<=m*H and i>=(m-1)*H+1:
                    A_matrix[n-H, i-((m-1)*H+1)] += window[m*H-i]*window[m*H-j]*ans_full[0,j]
                elif j<=m*H and j>=(m-1)*H+1:
                    A_matrix[n-H, j-((m-1)*H+1)] += window[m*H-i]*window[m*H-j]*ans_full[0,i]
                else:
                    b[n-H,0] = b[n-H,0] - window[m*H-i]*window[m*H-j]*ans_full[0,i]*ans_full[0,j]
        
        ######### Using qr factorization
        # print(m,':')
        # print(A_matrix)
        # print(b)
        Q,R = np.linalg.qr(A_matrix) # qr decomposition of A
        Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
        try:
            ans = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
        except:
            print('One sign meet singular matrix, please check original signal for detecting outliers')
            continue
            
#         #### fix error not implemented
        
#         for _ in range(n_iters):
#             Yw_real = torch.zeros((m+1, T//2+1), dtype=torch.float64)
#             Yw_imag = torch.zeros((m+1, T//2+1), dtype=torch.float64)
#             for m_temp in range(m+1):
#                 for n in range(T//2+1):
#                     for t in range(T):
#                         if m*H-t<0 or m*H-t>=N:
#                             continue
#                         Yw_real[m_temp,n] = Yw_real[m_temp,n]+ ans_full[0,t]*window[m*H-t]*np.cos(-2*np.pi*n*t/T)
#                         Yw_imag[m_temp,n] = Yw_imag[m_temp,n]+ ans_full[0,t]*window[m*H-t]*np.sin(-2*np.pi*n*t/T)
#             norm_vector = torch.sqrt(Yw_real**2 + Yw_imag**2)
#             b_Real = magnitude[m,:] * Yw_real/norm_vector
#             b_Imag = magnitude[m,:] * Yw_imag/norm_vector
        
#             b = np.concatenate((b_Real, b_Imag[1:T//2]))
#             # b shape: [1024]
           
#             result = np.linalg.solve(A,b)
#             ans[0,768:] = result [768:]
        
        ans_full[0,(m-1)*H+1:m*H+1] = ans[:,0]
    return ans_full



def reconstruct_from_Yw_with_H_2(magnitude, window, hop_length=2):
    
    # input window shape: [N]
    # input magnitude shape[(T+N-1)/H, T/2+1]
    # output shape: [1, T]
    T = (magnitude.shape[-1]-1)*2
    H = hop_length
    N = window.shape[0]
    L = Yw_to_L(magnitude, T,N,H)# L shape[(T+N-1)/H, T/2+1]
    # print(L)

    
    ans_full = np.zeros((1, T)).astype(np.float64)
    x0 = torch.sqrt(L[0,0])/window[0]
    
    x_hat_10 = x0 * window[2]
    x_hat_12 = L[1,2]/(x_hat_10)
    x_hat_11 = L[1,1]/(x_hat_10 + x_hat_12)
    x1 = x_hat_11/window[1]
    x2 = x_hat_12/window[0]
    
    ans_full[0,0] = x0
    ans_full[0,1] = x1
    ans_full[0,2] = x2
    
    for m in tqdm(range(2, L.shape[0]-3)):
        A_matrix = np.zeros((H+1, H), dtype=np.double)
        b = np.zeros((H+1,1), dtype=np.double)
        for n in range(H, 2*H):
            b[n-H,0] = L[m,n]
            for t in range(T):
                i=t
                j=(t+n)%T
                if m*H-i<0 or m*H-j<0 or m*H-i>=N or m*H-j>=N:
                    continue
                if i<=m*H and i>=(m-1)*H+1:
                    A_matrix[n-H, i-((m-1)*H+1)] += window[m*H-i]*window[m*H-j]*ans_full[0,j]
                elif j<=m*H and j>=(m-1)*H+1:
                    A_matrix[n-H, j-((m-1)*H+1)] += window[m*H-i]*window[m*H-j]*ans_full[0,i]
                else:
                    b[n-H,0] = b[n-H,0] - window[m*H-i]*window[m*H-j]*ans_full[0,i]*ans_full[0,j]
        
        ######### Using qr factorization
        # print(m,':')
        # print(A_matrix)
        # print(b)
        Q,R = np.linalg.qr(A_matrix) # qr decomposition of A
        Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
        try:
            ans = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
        except:
            print('One sign meet singular matrix, please check original signal for detecting outliers')
        ans_full[0,(m-1)*H+1:m*H+1] = ans[:,0]
        
    for m in range(L.shape[0]-3, L.shape[0]-2):
        remaining = T-1-(m-1)*H
        A_matrix = np.zeros((H+1, remaining), dtype=np.double)
        b = np.zeros((H+1,1), dtype=np.double)
        for n in range(H, 2*H):
            b[n-H,0] = L[m,n]
            for t in range(T):
                i=t
                j=(t+n)%T
                if m*H-i<0 or m*H-j<0 or m*H-i>=N or m*H-j>=N:
                    continue
                if i<=T-1 and i>=(m-1)*H+1:
                    A_matrix[n-H, i-((m-1)*H+1)] += window[m*H-i]*window[m*H-j]*ans_full[0,j]
                elif j<=T-1 and j>=(m-1)*H+1:
                    A_matrix[n-H, j-((m-1)*H+1)] += window[m*H-i]*window[m*H-j]*ans_full[0,i]
                else:
                    b[n-H,0] = b[n-H,0] - window[m*H-i]*window[m*H-j]*ans_full[0,i]*ans_full[0,j]
        
        ######### Using qr factorization
        # print(m,':')
        # print(A_matrix)
        # print(b)
        Q,R = np.linalg.qr(A_matrix) # qr decomposition of A
        Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
        try:
            ans = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
        except:
            print('One sign meet singular matrix, please check original signal for detecting outliers')
        ans_full[0,(m-1)*H+1:T] = ans[:,0]
    
    return ans_full





def reconstruct_from_Yw_with_H_3(magnitude, window, hop_length=3):
    
    # input window shape: [N]
    # input magnitude shape[(T+N-1)/H, T/2+1]
    # output shape: [1, T]
    T = (magnitude.shape[-1]-1)*2
    H = hop_length
    N = window.shape[0]
    L = Yw_to_L(magnitude, T,N,H)# L shape[(T+N-1)/H, T/2+1]
    # print(L)

    
    ans_full = np.zeros((1, T)).astype(np.float64)
    x0 = torch.sqrt(L[0,0])/window[0]
    x_hat_10 = (x0 * window[3]).numpy()
    C = x_hat_10
    x_hat_13 = (L[1,3]/(x_hat_10)).numpy()
    R_13 = L[1,3].numpy()
    R_12 = L[1,2].numpy()
    R_11 = L[1,1].numpy()
    R_10 = L[1,0].numpy()
    
    coeff = [-R_13/(C*C), C+R_12/C - R_13*R_13/(C**3), R_12*R_13/(C**2)-R_11]
    x_11_solutions = np.roots(coeff)
    
    minimum_error = 1e10
    minimum_x_hat_11 =0
    for x_hat_11 in x_11_solutions:
        x_hat_12 = (R_12-x_hat_11*x_hat_13)/x_hat_10
        temp_sum = x_hat_10**2 + x_hat_11**2 + x_hat_12**2 + x_hat_13**2
        if np.abs(temp_sum-R_10)<minimum_error:
            minimum_error = np.abs(temp_sum-R_10)
            minimum_x_hat_11 =x_hat_11
    if minimum_error>1e-10:
        print('rounding error' )
    
    x_hat_11 = minimum_x_hat_11
    x_hat_12 = (R_12-x_hat_11*x_hat_13)/x_hat_10
    
    ans_full[0,0] = x0
    ans_full[0,1] = x_hat_11/window[2]
    ans_full[0,2] = x_hat_12/window[1]
    ans_full[0,3] = x_hat_13/window[0]
    
    for m in tqdm(range(2, L.shape[0]-3)):
        A_matrix = np.zeros((H+1, H), dtype=np.double)
        b = np.zeros((H+1,1), dtype=np.double)
        for n in range(H, 2*H):
            b[n-H,0] = L[m,n]
            for t in range(T):
                i=t
                j=(t+n)%T
                if m*H-i<0 or m*H-j<0 or m*H-i>=N or m*H-j>=N:
                    continue
                if i<=m*H and i>=(m-1)*H+1:
                    A_matrix[n-H, i-((m-1)*H+1)] += window[m*H-i]*window[m*H-j]*ans_full[0,j]
                elif j<=m*H and j>=(m-1)*H+1:
                    A_matrix[n-H, j-((m-1)*H+1)] += window[m*H-i]*window[m*H-j]*ans_full[0,i]
                else:
                    b[n-H,0] = b[n-H,0] - window[m*H-i]*window[m*H-j]*ans_full[0,i]*ans_full[0,j]
        
        ######### Using qr factorization
        # print(m,':')
        # print(A_matrix)
        # print(b)
        Q,R = np.linalg.qr(A_matrix) # qr decomposition of A
        Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
        try:
            ans = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
        except:
            print('One sign meet singular matrix, please check original signal for detecting outliers')
        ans_full[0,(m-1)*H+1:m*H+1] = ans[:,0]
        
    for m in range(L.shape[0]-3, L.shape[0]-2):
        remaining = T-1-(m-1)*H
        A_matrix = np.zeros((H+1, remaining), dtype=np.double)
        b = np.zeros((H+1,1), dtype=np.double)
        for n in range(H, 2*H):
            b[n-H,0] = L[m,n]
            for t in range(T):
                i=t
                j=(t+n)%T
                if m*H-i<0 or m*H-j<0 or m*H-i>=N or m*H-j>=N:
                    continue
                if i<=T-1 and i>=(m-1)*H+1:
                    A_matrix[n-H, i-((m-1)*H+1)] += window[m*H-i]*window[m*H-j]*ans_full[0,j]
                elif j<=T-1 and j>=(m-1)*H+1:
                    A_matrix[n-H, j-((m-1)*H+1)] += window[m*H-i]*window[m*H-j]*ans_full[0,i]
                else:
                    b[n-H,0] = b[n-H,0] - window[m*H-i]*window[m*H-j]*ans_full[0,i]*ans_full[0,j]
        
        ######### Using qr factorization
        # print(m,':')
        # print(A_matrix)
        # print(b)
        Q,R = np.linalg.qr(A_matrix) # qr decomposition of A
        Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
        try:
            ans = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
        except:
            print('One sign meet singular matrix, please check original signal for detecting outliers')
        ans_full[0,(m-1)*H+1:T] = ans[:,0]
    
    return ans_full



def reconstruct_from_Yw_with_H_4(magnitude, window, hop_length=4):
    
    # input window shape: [N]
    # input magnitude shape[(T+N-1)/H, T/2+1]
    # output shape: [1, T]
    T = (magnitude.shape[-1]-1)*2
    H = hop_length
    N = window.shape[0]
    L = Yw_to_L(magnitude, T,N,H)# L shape[(T+N-1)/H, T/2+1]
    # print(L)

    
    ans_full = np.zeros((1, T)).astype(np.float64)
    
    x0 = torch.sqrt(L[0,0])/window[0]
    x_hat_10 = (x0 * window[4]).numpy()
    C = x_hat_10
    R_14 = L[1,4].numpy()
    R_13 = L[1,3].numpy()
    R_12 = L[1,2].numpy()
    R_11 = L[1,1].numpy()
    R_10 = L[1,0].numpy()
    x_hat_14 = R_14/(x_hat_10)
    D = C+R_14/C
    
    coeff = [R_14/(C*C*D) - R_14/(C*C*D)*R_14/(C*C),
             0 - R_13/(C*D) + R_14/(C*C*D)*R_13/C + R_13/(C*D)*R_14/(C*C),
             C + R_12/D - R_13/(C*D)*R_13/C - R_12/D*R_14/(C*C) - R_14/C*R_14/(C*C),
             R_12/D*R_13/C + R_13/C*R_14/C - R_11
            ]
    x_11_solutions = np.roots(coeff)
    
    minimum_error = 1e10
    minimum_x_hat_11 =0
    for x_hat_11 in x_11_solutions:
        x_hat_13 = R_13/C - R_14/(C*C)*x_hat_11
        x_hat_12 = (R_12 - R_13/C*x_hat_11 + R_14/(C*C)*x_hat_11*x_hat_11)/D
        temp_sum = x_hat_10**2 + x_hat_11**2 + x_hat_12**2 + x_hat_13**2 + x_hat_14**2
        if np.abs(temp_sum-R_10)<minimum_error:
            minimum_error = np.abs(temp_sum-R_10)
            minimum_x_hat_11 =x_hat_11
    if minimum_error>1e-10:
        print('rounding error' )
    
    x_hat_11 = minimum_x_hat_11
    x_hat_13 = R_13/C - R_14/(C*C)*x_hat_11
    x_hat_12 = (R_12 - R_13/C*x_hat_11 + R_14/(C*C)*x_hat_11*x_hat_11)/D
    
    ans_full[0,0] = x0
    ans_full[0,1] = x_hat_11/window[3]
    ans_full[0,2] = x_hat_12/window[2]
    ans_full[0,3] = x_hat_13/window[1]
    ans_full[0,4] = x_hat_14/window[0]
    
    
    
    for m in tqdm(range(2, L.shape[0]-3)):
        A_matrix = np.zeros((H+1, H), dtype=np.double)
        b = np.zeros((H+1,1), dtype=np.double)
        for n in range(H, 2*H):
            b[n-H,0] = L[m,n]
            for t in range(T):
                i=t
                j=(t+n)%T
                if m*H-i<0 or m*H-j<0 or m*H-i>=N or m*H-j>=N:
                    continue
                if i<=m*H and i>=(m-1)*H+1:
                    A_matrix[n-H, i-((m-1)*H+1)] += window[m*H-i]*window[m*H-j]*ans_full[0,j]
                elif j<=m*H and j>=(m-1)*H+1:
                    A_matrix[n-H, j-((m-1)*H+1)] += window[m*H-i]*window[m*H-j]*ans_full[0,i]
                else:
                    b[n-H,0] = b[n-H,0] - window[m*H-i]*window[m*H-j]*ans_full[0,i]*ans_full[0,j]
        
        ######### Using qr factorization
        # print(m,':')
        # print(A_matrix)
        # print(b)
        Q,R = np.linalg.qr(A_matrix) # qr decomposition of A
        Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
        try:
            ans = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
        except:
            print('One sign meet singular matrix, please check original signal for detecting outliers')
        ans_full[0,(m-1)*H+1:m*H+1] = ans[:,0]
        
    for m in range(L.shape[0]-3, L.shape[0]-2):
        remaining = T-1-(m-1)*H
        A_matrix = np.zeros((H+1, remaining), dtype=np.double)
        b = np.zeros((H+1,1), dtype=np.double)
        for n in range(H, 2*H):
            b[n-H,0] = L[m,n]
            for t in range(T):
                i=t
                j=(t+n)%T
                if m*H-i<0 or m*H-j<0 or m*H-i>=N or m*H-j>=N:
                    continue
                if i<=T-1 and i>=(m-1)*H+1:
                    A_matrix[n-H, i-((m-1)*H+1)] += window[m*H-i]*window[m*H-j]*ans_full[0,j]
                elif j<=T-1 and j>=(m-1)*H+1:
                    A_matrix[n-H, j-((m-1)*H+1)] += window[m*H-i]*window[m*H-j]*ans_full[0,i]
                else:
                    b[n-H,0] = b[n-H,0] - window[m*H-i]*window[m*H-j]*ans_full[0,i]*ans_full[0,j]
        
        ######### Using qr factorization
        # print(m,':')
        # print(A_matrix)
        # print(b)
        Q,R = np.linalg.qr(A_matrix) # qr decomposition of A
        Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
        try:
            ans = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
        except:
            print('One sign meet singular matrix, please check original signal for detecting outliers')
        ans_full[0,(m-1)*H+1:T] = ans[:,0]
    
    return ans_full