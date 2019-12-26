import json
import pandas as pd
import numpy as np
import gc
import datetime
import re
import seaborn as sns
import matplotlib.pyplot as plt

def nmf(X, k, max_iter, epsilon, l_rate=1, use_l_rate=False):
    '''
    Non-negative matrix factorization with multiplicative updates
    Inputs:
        X: (numpy array)matrix
        k: (int)dimension to keep
        max_iter: (int) the maximum number of iteration
        epsilon: the threshold to stop iteration
        l_rate: learning rate specified
        use_l_rate: if True, use the arbitrary learning rate
    Outputs:
        two decomposition matrix: W & H
    '''
    U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    U[np.where(U <= 0)] = -U[np.where(U <= 0)]
    Vt[np.where(Vt <= 0)] = -Vt[np.where(Vt <= 0)]
        
    
    init_W = U[:, :k].dot(np.diag(sigma[:k]))
    init_W[init_W <= 0] = 0
    init_H = Vt[:k, :]
    init_H[init_H <= 0] = 0
    
    W = init_W
    H = init_H
    W_gradient = - X.dot(H.T) + W.dot(H).dot(H.T)
    H_gradient = - W.T.dot(X) + W.T.dot(W).dot(H)
    H_r = np.divide(H, (W.T.dot(W).dot(H)))
    W_r = np.divide(W, (W.dot((H.dot(H.T)))))

    for i in np.arange(0, max_iter, 1):
        
        if use_l_rate:
            W = W - l_rate * W_gradient
        else:
            W = W - np.multiply(W_r, W_gradient)
            W_r = np.divide(W, (W.dot(H).dot(H.T)))

        W_gradient = - X.dot(H.T) + W.dot(H).dot(H.T)
        W[np.where(W <= 0)] = init_W[np.where(W <= 0)]
        
        if use_l_rate:
            H = H - l_rate * H_gradient
        else:
            H = H - np.multiply(H_r, H_gradient)
            H_r = np.divide(H, (W.T.dot(W).dot(H)))

        H_gradient = - W.T.dot(X) + W.T.dot(W).dot(H)
        H[np.where(H <= 0)] = init_H[np.where(H <= 0)]

        if np.linalg.norm(X - W.dot(H)) < epsilon:
            print("the stopping condition is reached, the estimated matrix is similar enough to the origin")
            print("the number of iteration is {}".format(i))
            break
    
    print("the max iteration is reached")
    return W, H


def nmf_two_block(X, k, max_iter, epsilon, l_rate=1, use_l_rate=False):
    '''
    Non-negative matrix factorization with multiplicative updates
    using two block gradient descent algorithm
    Inputs:
        X: (numpy array)matrix
        k: (int)dimension to keep
        max_iter: (int) the maximum number of iteration
        epsilon: the threshold to stop iteration
        l_rate: learning rate specified
        use_l_rate: if True, use the arbitrary learning rate
    Outputs:
        two decomposition matrix: W & H
    '''
    U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    U[np.where(U <= 0)] = -U[np.where(U <= 0)]
    Vt[np.where(Vt <= 0)] = -Vt[np.where(Vt <= 0)]
        
    
    init_W = U[:, :k].dot(np.diag(sigma[:k]))
    init_W[init_W <= 0] = 0
    init_H = Vt[:k, :]
    init_H[init_H <= 0] = 0
    
    W = init_W
    H = init_H
    W_gradient = - X.dot(H.T) + W.dot(H).dot(H.T)
    H_gradient = - W.T.dot(X) + W.T.dot(W).dot(H)
    H_r = np.divide(H, (W.T.dot(W).dot(H)))
    W_r = np.divide(W, (W.dot((H.dot(H.T)))))
    
    for i in np.arange(0,max_iter,1):
        if i == 0:
            first_norm = np.linalg.norm(W - (W - np.multiply(W_r, W_gradient)))
            #print(first_norm)
        current_norm = np.linalg.norm(W - (W - np.multiply(W_r, W_gradient)))
        #print(current_norm)
        
        if use_l_rate:
            W = W - l_rate * W_gradient
        else:
            W = W - np.multiply(W_r, W_gradient)
            W_r = np.divide(W, (W.dot(H).dot(H.T)))

        W_gradient = - X.dot(H.T) + W.dot(H).dot(H.T)
        W[np.where(W <= 0)] = init_W[np.where(W <= 0)]
        
        if current_norm <= epsilon * first_norm:
            print("the stopping condition is reached, the estimated matrix is similar enough to the origin")
            print("the number of iteration is {}".format(i))
            break
                        
    for j in range(max_iter):
        if i == 0:
            first_norm = np.linalg.norm(H - (H - np.multiply(H_r, H_gradient)))
        current_norm = np.linalg.norm(H - (H - np.multiply(H_r, H_gradient)))
        
        if use_l_rate:
            H = H - l_rate * H_gradient
        else:
            H = H - np.multiply(H_r, H_gradient)
            H_r = np.divide(H, (W.T.dot(W).dot(H)))
            
        H_gradient = - W.T.dot(X) + W.T.dot(W).dot(H)
        H[np.where(H <= 0)] = init_H[np.where(H <= 0)]

        if current_norm <= epsilon * first_norm:
            print("the stopping condition is reached, the estimated matrix is similar enough to the origin")
            print("the number of iteration is {}".format(i))
            break
    
    print("the max iteration is reached")
    return W, H