import json
import pandas as pd
import numpy as np
import gc
import datetime
import re
import seaborn as sns
import matplotlib.pyplot as plt

#################################################################################
##                           For Classifiers                                   ##
#################################################################################
def truncated_svd(X, k):
    '''
    Decompose matrix using truncated SVD
    Inputs:
        X: input matrix
        k: dimension to keep
    Outputs:
        new matrix constructed by truncated svd
    '''
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    S_k = S[:k]
    S_mat = np.diag(S_k)
    
    return U_k.dot(S_mat).dot(Vt_k)


def get_W(X):
    '''
    The function is used to calculate matrix W and H from
    the input matrix using Non-negative Factorization.
    Inputs: complete matrix
    Return: W, H matrix
    '''
    print("nmf (two block)...")
    W, H = nmf_two_block(X, 10, 100000, 10**(-6))
    
    return W, H


def least_square_regularized(X, y, k, lamb, reg=True, full=False):
    '''
    This function is used to calculate the weight with different setting
    Inputs:
        X: dataframe
        y: label
        k: k dimension in SVD setting
        lamb: regularization parameters for ridge regression
        reg: (boolean) using ridge regresison or not
        full: use the full matrix after SVD or not
    Return: weights vector
    '''
    if full:
        U_k, sigma_k, Vt_k = np.linalg.svd(X, full_matrices=False)
    
    else:
        U_k, sigma_k, Vt_k = dimensional_reduction(X, k, y, allm=True, 
                                                   get_weight=False)
    
    if reg:
        inv_ele = np.linalg.inv((np.diag(sigma_k).T).dot(
                                np.diag(sigma_k)) + 
                                lamb * np.diag(np.repeat(1, np.diag(sigma_k).shape[0])))

        rweight = np.array(((Vt_k.T.dot(inv_ele)).dot(
                             np.diag(sigma_k).T)).dot(U_k.T)).dot(y)
    
    else:
        pinv = ((np.diag(sigma_k)).T).dot(
                 np.linalg.inv(np.diag(sigma_k).dot((np.diag(sigma_k)).T)))
        return Vt_k.T.dot(pinv).dot(U_k.T).dot(y)
        
    return rweight


###################################################################################
##                      Model Selection Preprocessing                            ##
###################################################################################
def combination_indexer(var_len, index_list):
    '''
    This function is used to provide binary index for the list of variables
    which will be used in get_var_combinations function
    Inputs:
        var_len: the length of the variable list
        index_list: list of the indices
    Returns: a list with all possible binary combinations (list of list)
    '''
    if len(index_list) == var_len:
        return [index_list]
    
    else:
        total_list = []
        for i in range(2):
            new_list = index_list + [i]
            total_list += combination_indexer(var_len, new_list)
    
    return total_list


def get_var_combinations(var_list):
    '''
    This function is used to provide all possible combinations of variables
    which will be used in the input of machine learning methods
    Inputs:
        var_list: a list of variables
    Returns: list of all possible combinations of variables in the given
             variable list (list of list)
    '''
    index_combination = combination_indexer(len(var_list), [])
    var_combinations = []
    
    for combination in index_combination:
        var_sublist = []
        for i, val in enumerate(combination):
            if val == 1:
                var_sublist.append(var_list[i])

        var_combinations.append(var_sublist)
    
    return var_combinations


def get_limited_permuted_indices(permuted_index_list, k):
    '''
    The function is used to provided the limit length of
    the index permutation
    Inputs: 
        permuted_index_list: permuted index list
        k: length of column indices selected
    Return: limited index list
    '''
    permuted_index_list_limited = []
    for i in get_var_combinations(np.arange(0,10,1))[1:]:
        if len(i) <= k:
            permuted_index_list_limited.append(i)
    return permuted_index_list_limited


################################################################################
##                          Model Selection                                   ##
################################################################################
def best_representative_model(W, H, y, permuted_index_list, full_indices):
    '''
    The function is used to compare different predicted model (Ridge Regression)
    with different column indices. We only select the row indices 
    with non-null y.
    Inputs:
        W: W matrix
        H: H matrix
        y: y vector with complete values
        permuted_index_list: list of indices combination for slicing
        full_indices: row indices with complete value in y
    Return: tuples of best indices, average squared error and weights
    '''
    #W, H = get_W(X)
    best_error = np.inf
    
    for idx in permuted_index_list:
        #print("computing permutation..." + str(idx))

        Wn = W[full_indices][:, idx]
        z = least_square_regularized(Wn, y, 10, 0.01, reg=True, full=True)
        y_hat = Wn.dot(z)
        
        error = sum((y - y_hat) ** 2) / len(y)
        if error < best_error:
            best_error = error
            best_indices = idx
            best_weights = z

    return (best_indices, best_error, best_weights)


def model_k_selection(k_list, y_list, y_names, permuted_index_list):
    '''
    The function is used to implement the complete operation of different
    column index length k with rating of different business type.
    Inputs:
        k_list: list of k values
        y_list: average rating for Italian, Mexican and Chinese business
        y_names: Name for the final dictionary
        permuted_index_list: list of column indices for slicing
    Return: performance dictionary
    '''
    y_performances = {}
    idx = 0
    models_list = y_names
    for y in y_list:
        full_indices = list(np.where(~np.isnan(y))[0])
        k_dict = {}
        for k in k_list:
            permuted_index_list_limited = get_limited_permuted_indices(permuted_index_list, k)
            perf_tuple = best_representative_model(W, H, y[full_indices, :], permuted_index_list_limited, full_indices)
            k_dict[k] = perf_tuple
        y_performances[models_list[idx]] = k_dict
        idx += 1
    
    return y_performances


def plot_k_model_selection(k_list, y_list, y_names, permuted_index_list):
    '''
    The function is used to plot the result of model_k_selection function
    '''
    y_performances = model_k_selection(k_list, y_list, y_names, permuted_index_list)
    
    error_array_list = []
    for i in np.arange(0, len(y_list), 1):
        error_list = []
        for idx, k in enumerate(k_list):
            error_list.append(y_performances[y_names[i]][k][1][0])
        
        error_array_list.append(error_list)
        
    for error_list in error_array_list:
        plt.plot(np.arange(0, len(error_list), 1), error_list)
    
    return error_array_list