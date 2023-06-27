import IPython.display as ipd

import numpy as np
import torch


# import stft_64_pad_0 as stft
import stft_64 as stft
from audio_processing import griffin_lim

from scipy.io.wavfile import read
import time

def overdetermined_linear_system_solver(ori_A, ori_b, threshold = 1e-13):
    # solve Ax = b
    # A shape:[m,n]
    # x shape:[n,1]
    # b shape:[m,1] or [m,]
    # output: x, numpy array, shape [n,1]
    (m,n) = ori_A.shape
    A = np.zeros((m,n), dtype= np.float64)
    A[:,:] = ori_A[:,:]
    b = np.zeros((m,1), dtype= np.float64)
    if len(ori_b.shape)==2:
        b[:,:] = ori_b[:,:]
    else:
        b[:,0] = ori_b[:]
    
    remaining_row = [i for i in range(m)]
    remaining_column = [i for i in range(n)]
    idx = 0 
    existing_column = []
    while(idx<n):
        # if idx %10==0:
        #     print('\r%d/%d'%(idx,n), end="")
        # print(A)
        # print(b)
        maxabs_value = 0
        maxabs_ij=(0,0)
        for i in remaining_row:
            for j in remaining_column:
                if np.abs(A[i,j])>maxabs_value:
                    maxabs_ij = (i,j)
                    maxabs_value = np.abs(A[i,j])
        if maxabs_value<threshold:
            print('Underdeterminant')
            return None
        
        select_row = maxabs_ij[0]
        select_column = maxabs_ij[1]
        for i in range(m):
            if i==select_row:
                continue
            minus_cof = A[i,select_column]/A[select_row, select_column]
            for j in range(n):
                A[i,j] = A[i,j] - minus_cof*A[select_row, j]
            b[i] = b[i] - minus_cof*b[select_row]
        remaining_column.remove(select_column)
        remaining_row.remove(select_row)
        # print(remaining_column)
        existing_column.append((select_row,select_column))
        idx+=1
    print('\n')
    # print(A)
    # print(b)
    x = np.zeros((n,1), dtype= np.float64)
    idx = n-1
    while(idx>=0):
        select_row, select_column = existing_column[idx]
        x[select_column,0] = b[select_row,0]/A[select_row, select_column]
        # for i in range(m):
        #     b[i,0] = b[i,0] - A[i, select_column]*x[select_column,0]
        #     A[i, select_column] = 0
        idx = idx-1
        # print('A',A)
        # print('x',x)
        # print('b',b)
    return x
    


class hop_7_solver():
    #magnitude shape: [1, channels, 8]
    #[1, channels, 2] for [1, hop_length*0: hop_length*4]
    #[1, channels, 3] for [1, hop_length*1: hop_length*5]
    #[1, channels, 4] for [1, hop_length*2: hop_length*6]
    #[1, channels, 5] for [1, hop_length*3: hop_length*7]
    #x: [1, hop_length*7]
    
    def __init__(self, forward_basis, hop_length, channels, win_length):
        self.forward_basis = forward_basis
        self.hop_length = hop_length
        self.channels = channels
        self.win_length = win_length
        
    def test(self, p, magnitude):
        hop_iter = 0
        start_frame = hop_iter * self.hop_length
        M_Rc = torch.sum(self.forward_basis[:self.channels,0,:] * p[:,start_frame: start_frame + self.win_length], dim =1)
        M_Ic = torch.sum(self.forward_basis[self.channels:,0,:] * p[:,start_frame: start_frame + self.win_length], dim =1)
        M_c_square = torch.sqrt(M_Rc**2+M_Ic**2) - self.magnitude[0,:,hop_iter]
        # M_c_square = torch.abs(torch.sqrt(M_Rc**2+M_Ic**2) - self.magnitude[0,:,hop_iter])
        M_c_square = M_c_square.unsqueeze(0).unsqueeze(2)
        
        # print(M_c_square.shape)
        for hop_iter in range(1,4):
            start_frame = hop_iter * self.hop_length
            M_Rc = torch.sum(self.forward_basis[:self.channels,0,:] * p[:,start_frame: start_frame + self.win_length], dim =1)
            M_Ic = torch.sum(self.forward_basis[self.channels:,0,:] * p[:,start_frame: start_frame + self.win_length], dim =1)
            M_c_square_temp = torch.sqrt(M_Rc**2+M_Ic**2) - self.magnitude[0,:,hop_iter]
            # M_c_square_temp = torch.abs(torch.sqrt(M_Rc**2+M_Ic**2) - self.magnitude[0,:,hop_iter])
            M_c_square_temp = M_c_square_temp.unsqueeze(0).unsqueeze(2)
            M_c_square = torch.cat((M_c_square, M_c_square_temp), dim=2)
        return M_c_square.reshape(-1)
    
    
    def func(self, p):
        hop_iter = 0
        start_frame = hop_iter * self.hop_length
        M_Rc = torch.sum(self.forward_basis[:self.channels,0,:] * p[:,start_frame: start_frame + self.win_length], dim =1)
        M_Ic = torch.sum(self.forward_basis[self.channels:,0,:] * p[:,start_frame: start_frame + self.win_length], dim =1)
        M_c_square = torch.sqrt(M_Rc**2+M_Ic**2) - self.magnitude[0,:,hop_iter]
        # M_c_square = torch.abs(torch.sqrt(M_Rc**2+M_Ic**2) - self.magnitude[0,:,hop_iter])
        M_c_square = M_c_square.unsqueeze(0).unsqueeze(2)
        
        # print(M_c_square.shape)
        for hop_iter in range(1,4):
            start_frame = hop_iter * self.hop_length
            M_Rc = torch.sum(self.forward_basis[:self.channels,0,:] * p[:,start_frame: start_frame + self.win_length], dim =1)
            M_Ic = torch.sum(self.forward_basis[self.channels:,0,:] * p[:,start_frame: start_frame + self.win_length], dim =1)
            M_c_square_temp = torch.sqrt(M_Rc**2+M_Ic**2) - self.magnitude[0,:,hop_iter]
            # M_c_square_temp = torch.abs(torch.sqrt(M_Rc**2+M_Ic**2) - self.magnitude[0,:,hop_iter])
            M_c_square_temp = M_c_square_temp.unsqueeze(0).unsqueeze(2)
            M_c_square = torch.cat((M_c_square, M_c_square_temp), dim=2)
        return M_c_square.reshape(-1)
    
    def solve(self, magnitude, initial_guess, n_iters=50, lambda_JTJ=1):
        #magnitude shape:      [1, hop_length*2+1, 4]
        #initial_guess shape:  [1, 7*hop_length]
        
        self.magnitude =magnitude
        start_time = time.time()
        for i in range(n_iters):
            print('\rIter %d/%d: Used times: %.2f' %(i,n_iters,time.time()-start_time), end="")
            # check(recon)
            #print('#')
            x = torch.tensor(initial_guess.detach().numpy(), dtype=torch.float64, requires_grad = True) 
            # x = torch.from_numpy(np.zeros(initial_guess.shape))
            # x[:,:] = initial_guess[:,:]
            # x = torch.DoubleTensor(x)
            
            J = torch.autograd.functional.jacobian(self.func, x)
            # print('J shape', J.shape)
           
            J = J.squeeze(1)
            # print('')
            # print('J',J)
            target = self.func(x).detach().numpy()
            
            # Q, R = np.linalg.qr(J, mode='reduced')
            # # print(Q, R)
            # Qb = np.matmul(Q.T, target)
            # minus = np.linalg.solve(R,Qb)
            
            temp = J.T @ J
            minus = (torch.inverse(temp + lambda_JTJ * torch.diag(torch.diag(temp, 0))))@ (J.T @ target)
            # print('res', minus.T)
            # print('check', torch.matmul(J, minus))
            # print('error', target)
            minus = minus.numpy()
            # print(J.shape)
            # print(x.shape)
            # print(self.func(x).shape)
            # minus = overdetermined_linear_system_solver(J.numpy(), self.func(x).numpy())
            # minus = torch.from_numpy(minus)
        
            initial_guess = initial_guess - minus.T
            # if torch.sum(torch.abs(norm_vector/1000-magnitude[0,:]))<1e-10:
            #     break
            
        return initial_guess
    
    

class get_audio_from_spectrogram:
    
    def __init__(self, stft_fn):
        self.stft_fn = stft_fn
        self.hop_length = stft_fn.hop_length
        self.win_length = 4 * self.hop_length
        self.channels = 2 * self.hop_length + 1
        # c=0,.....,255 in equation (4) could not form a full-rank matrix!
        # we use c=0,2,4,6,...,510 to get the full-rank(256) matirx and inverse
        
        self.cos_matrix = np.zeros((self.hop_length,self.hop_length), dtype=np.double)
        for c in range(self.hop_length):
            for i in range(self.hop_length):
                self.cos_matrix[i,c] = np.cos(2*np.pi*(2*c)*i/(self.hop_length * 4), dtype=np.double)
                
        # compute inverse of matrix COS
        self.inv_coefficient_solver = np.linalg.inv(self.cos_matrix)
#         self.inv_coefficient_solver = np.zeros((256,256), dtype=np.double)
#         for i in range(1,256):
#             self.inv_coefficient_solver[i,0]=(i%2)/128
#         for j in range(1,256):
#             self.inv_coefficient_solver[0,j]=(j%2)/128
    
#         for i in range(1,256):
#             for j in range(1,256):
#                 self.inv_coefficient_solver[i,j]=(np.cos(np.pi*i*j/256) + 2*((i+j)%2) -1)/128
        
        # compute how to use inverse matrix to get coefficient a-matrix
        # target_matrix: c=1,3,5,7,9,....,511,512
        self.target_matrix = np.zeros((self.hop_length,self.hop_length+1), dtype=np.double)
        for c in range(self.hop_length):
            for i in range(self.hop_length):
                self.target_matrix[i,c] = np.cos(2*np.pi*(2*c+1)*i/(self.hop_length*4), dtype=np.double)
        for i in range(self.hop_length):
            self.target_matrix[i,self.hop_length] = np.cos(2*np.pi*(self.hop_length*2)*i/(self.hop_length*4), dtype=np.double)
        self.coefficient_a_matrix = np.matmul(self.inv_coefficient_solver, self.target_matrix)  #shape [256,257]
        
    def calculate_part_audio(self, magnitude, prev768):
        # magnitude shape: [1, 513]
        # prev_768  shape: [1, 768]
        # output: guess next 256. shape [256]
        
        R_Tc = torch.sum(self.stft_fn.forward_basis[:self.channels,0,0:self.win_length - self.hop_length]  * prev768, dim=1)
        I_Tc = torch.sum(self.stft_fn.forward_basis[self.channels:,0,0:self.win_length - self.hop_length]  * prev768, dim=1)
        M_Tc = magnitude[0,:]
        # R_Tc, I_Tc, M_Tc shape: [513]
        
        Constant_origin_equation = R_Tc**2 + I_Tc**2 - M_Tc**2
        Constant_origin_equation = Constant_origin_equation.unsqueeze(1)
        # Constant shape: [513, 1]
        
        A_origin_equation = 2 * R_Tc.unsqueeze(1) * self.stft_fn.forward_basis[:self.channels,0,self.win_length - self.hop_length:] + \
                            2 * I_Tc.unsqueeze(1) * self.stft_fn.forward_basis[self.channels:,0,self.win_length - self.hop_length:]
        # A shape: [513, 256]
        
        base_constant = Constant_origin_equation[:self.channels - 1:2,:] #shape [256,1]
        base_A        = A_origin_equation[:self.channels - 1:2,:]        #shape [256,256]
        
        constant = np.zeros((self.hop_length+1,1), dtype=np.double)
        A        = np.zeros((self.hop_length +1 ,self.hop_length), dtype=np.double) 
        constant[:self.hop_length,:] = Constant_origin_equation[1:self.channels - 1:2,:]
        A[:self.hop_length,:]        = A_origin_equation[1:self.channels - 1:2,:]
        constant[self.hop_length,:] = Constant_origin_equation[self.channels - 1,:]
        A[self.hop_length,:]        = A_origin_equation[self.channels - 1,:]
        
        constant = constant - np.matmul(self.coefficient_a_matrix.T, base_constant.numpy())
        A        = A        - np.matmul(self.coefficient_a_matrix.T, base_A.numpy())
#         test_result= constant + np.matmul(A,test.T)
#         print(constant)
#         print(test_result.T)
#         print(test)
        
        # solve constant + Ax=0
        # solve Ax= -constant
        Q, R = np.linalg.qr(A, mode='reduced')
        #print(Q.shape, R.shape)
        Qb = np.matmul(Q.T,-constant)
        result = np.linalg.solve(R,Qb)
        
        # result shape: [256,1]
        return result
    
    def fix_error(self, magnitude, recon, n_iters=1000):
        # magnitude shape: [1, 513]
        # recon     shape: [1, 1024]
        # output: guess next 256. shape [256]
        
        # solve Ax= b
        a=[]
        for i in range(self.channels):
            a.append(i)
        for i in range(self.channels+1,self.win_length+1):
            a.append(i)
        
        A = self.stft_fn.forward_basis[a,0,:].numpy()
        
        for _ in range(n_iters):
            Real = torch.sum(self.stft_fn.forward_basis[:self.channels,0,0:self.win_length]  * recon, dim=1)
            Imag = torch.sum(self.stft_fn.forward_basis[self.channels:,0,0:self.win_length]  * recon, dim=1)
            Real *= 1000
            Imag *= 1000
            # Real shape: [513]
            norm_vector = torch.sqrt(Real**2 + Imag**2)
            b_Real = magnitude[0,:] * Real/norm_vector
            b_Imag = magnitude[0,:] * Imag/norm_vector
        
            b = np.concatenate((b_Real, b_Imag[1:self.channels-1]))
            # b shape: [1024]
           
            result = np.linalg.solve(A,b)
            recon[0,self.win_length - self.hop_length:] = result [self.win_length - self.hop_length:]
            # if torch.sum(torch.abs(norm_vector/1000-magnitude[0,:]))<1e-10:
            #     break
            
        return result[self.win_length - self.hop_length:]
    
    def fix_error_newton(self, magnitude, recon, n_iters=50):
        # magnitude shape: [1, 513]
        # recon     shape: [1, 1024]
        # output: guess next 256. shape [256]
        
        # solve Ax= b
        a=[]
        for i in range(self.channels):
            a.append(i)
        for i in range(self.channels+1,self.win_length+1):
            a.append(i)
        A = self.stft_fn.forward_basis[a,0,:].numpy()
        A_inverse = np.linalg.inv(A)
        A_inverse = torch.from_numpy(A_inverse)
        recon768 = torch.from_numpy(recon[:,:self.win_length - self.hop_length].T)
        
        def func(x):
            Real = torch.matmul(self.stft_fn.forward_basis[:self.channels,0,:self.win_length - self.hop_length], recon768) + \
                   torch.matmul(self.stft_fn.forward_basis[:self.channels,0,self.win_length - self.hop_length:], x)
            
            Imag = torch.matmul(self.stft_fn.forward_basis[self.channels:,0,:self.win_length - self.hop_length], recon768) + \
                   torch.matmul(self.stft_fn.forward_basis[self.channels:,0,self.win_length - self.hop_length:], x)
            # Real shape: [513,1]
            norm_vector = torch.sqrt(Real**2 + Imag**2)
            b = magnitude.squeeze(0) - norm_vector.squeeze(1)
            return b
        
        def f(x):
            Real = torch.matmul(self.stft_fn.forward_basis[:self.channels,0,:self.win_length - self.hop_length], recon768) + \
                   torch.matmul(self.stft_fn.forward_basis[:self.channels,0,self.win_length - self.hop_length:], x)
            
            Imag = torch.matmul(self.stft_fn.forward_basis[self.channels:,0,:self.win_length - self.hop_length], recon768) + \
                   torch.matmul(self.stft_fn.forward_basis[self.channels:,0,self.win_length - self.hop_length:], x)
            # Real shape: [513,1]
            norm_vector = torch.sqrt(Real**2 + Imag**2)
            b = magnitude.squeeze(0) - norm_vector.squeeze(1)
            return b
        
        def check(x):
            Real = torch.sum(self.stft_fn.forward_basis[:self.channels,0,0:self.win_length]  * x, dim=1)
            Imag = torch.sum(self.stft_fn.forward_basis[self.channels:,0,0:self.win_length]  * x, dim=1)
            norm_vector = torch.sqrt(Real**2 + Imag**2)
            print(torch.sum(torch.abs(magnitude[0,:] - norm_vector)))
        
        for _ in range(n_iters):
            # check(recon)
            #print('#')
            x = torch.from_numpy(recon[:,self.win_length - self.hop_length:])
            x = torch.DoubleTensor(x)
            
            J = torch.autograd.functional.jacobian(func, x.T)
            #print(J.shape)
           
            J = J.squeeze(2)
            
            Q, R = np.linalg.qr(J, mode='reduced')
            #print(Q.shape, R.shape)
            Qb = np.matmul(Q.T,f(x.T))
            minus = np.linalg.solve(R,Qb)
            
            recon[0,self.win_length - self.hop_length:] = recon[0, self.win_length - self.hop_length:] - minus
            # if torch.sum(torch.abs(norm_vector/1000-magnitude[0,:]))<1e-10:
            #     break
            
        return recon[0,self.win_length - self.hop_length:]