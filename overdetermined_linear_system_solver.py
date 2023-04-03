import numpy as np


def overdetermined_linear_system_solver(ori_A, ori_b, threshold = 1e-12):
    # solve Ax = b
    # A shape:[m,n]
    # x shape:[n,1]
    # b shape:[m,1]
    # output: x, numpy array, shape [n,1]
    (m,n) = ori_A.shape
    A = np.zeros((m,n), dtype= np.float64)
    A[:,:] = ori_A[:,:]
    b = np.zeros((m,1), dtype= np.float64)
    b[:,:] = ori_b[:,:]
    
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
    