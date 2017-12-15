import numpy as np
def PCA(data,dimensions):
    '''
    data is the original data set，rows are samples of data,columns are the features
    '''

    # making data zero-means
    average = np.mean(data,0)
    data_zero = np.mat(data-average)
    
    #covariance
    covariance = np.cov(data_zero,rowvar=False)
    
    #eigenvalues
    eig_var,eig_vec = np.linalg.eig(covariance)
    
    
    #from the numpy doc, the eig_var may not be ordered.
    sort_eig = np.argsort(-eig_var)
    #return the index that make a sorted array

    #so we got the sorted eig_var
    sort_eig = sort_eig[:dimensions]
    principal_vec = np.mat(eig_vec[:,sort_eig])
    
    return principal_vec, np.dot(data_zero, principal_vec), average
