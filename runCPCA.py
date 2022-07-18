import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import scipy.stats as stats
import math
import scipy.io
#RunCPCA 1st cell
####  may need to turns NaN into nan, which could be identified by matlab
def runCPCA(Z, G, H, n, varamaxFlag):      
# X - rows are observations (Points), columns are features (variables)
# X should be zscored along the columns
# G - Predictor variable matrix along 1st dim -rows
# H - Predictor variable matrix along 2nd dim - cols
# n - Select top n components
    varamaxFlag=False
    N,NNN = np.shape(Z)    ########deal with size, no of rows in Z  
    if  np.size(G) != 1 and H == 1:  ###one of GH are 1
        C = la.pinv(G.T@G)@G.T@Z # Linear regression
        S = G@C
        E = Z - S
    elif G == 1 and np.size(H) != 1:
        B = Z@H@la.pinv(np.transpose(H)@H) # Linear regression
        S = B@np.transpose(H)
        E = Z - S
    elif G == 1 and H == 1:
        S = Z
        E = Z - S
    OutStruct = {}
    temp = np.corrcoef(np.transpose(stats.zscore(S,ddof=1)))
    temp = (np.nan_to_num(temp, nan=0.0))
    [OutStruct['EigValGH'],OutStruct['EigVecGH']] = la.eig(temp)
    OutStruct['EigValGH'] = sorted((OutStruct['EigValGH']),reverse = True)
    [OutStruct['UGH'], OutStruct['DGH'], OutStruct['VGH']] = la.svd(S,0) #svd for GC,0 for 'econ' mode   ###### the third output of la.svd in python is transposed compared with it in matlab    ######,'econ'eliminated
    OutStruct['VGH']=np.transpose(OutStruct['VGH']) #need a transpose to get the same result as matlab
    OutStruct['DGH'] = np.diag(OutStruct['DGH'])
    ######transpose the 3rd output
    [OutStruct['loadings_VD_sqrtN_GH'], OutStruct['scores_NU_GH'], OutStruct['loadings_VD_sqrtN_GH_inv'], OutStruct['T']] = findLoadingsScores(OutStruct['UGH'], OutStruct['DGH'], OutStruct['VGH'], N, n, varmaxFlag = False)
    
    if  np.size(G) != 1 or H != 1:
        temp = np.corrcoef(np.transpose(stats.zscore(E,ddof=1)))
        temp = (np.nan_to_num(temp, nan=0.0))
        [OutStruct['EigValE'],OutStruct['EigVecE']] = la.eig(temp)
        OutStruct['EigValE'] = sorted((OutStruct['EigValE']),reverse = True)
        [OutStruct['UE'], OutStruct['DE'], OutStruct['VE']] = la.svd(E,0) #  SVD for E, 0 for 'econ' mode
        OutStruct['VE']=np.transpose(OutStruct['VE']) #need a transpose to get the same result as matlab
        OutStruct['DE'] = np.diag(OutStruct['DE'])
    
    OutStruct['Z'] = Z
    OutStruct['G'] = G
    OutStruct['H'] = H
    OutStruct['E'] = E
    OutStruct['S'] = S
    return OutStruct

#RunCPCA 2nd cell
def findLoadingsScores(U, D, V, N, n, varmaxFlag):
    loadings_VD_sqrtN = V@np.diag(np.diag(D))/np.sqrt(N-1)
    scores_NU = np.sqrt(N-1)*U
    loadings_VD_sqrtN_inv = np.transpose(la.pinv(loadings_VD_sqrtN))
    loadings_VD_sqrtN = loadings_VD_sqrtN[:,0:n]
    loadings_VD_sqrtN_inv = loadings_VD_sqrtN_inv[:,0:n]
    scores_NU = scores_NU[:,0:n]
    T = np.nan

    return loadings_VD_sqrtN, scores_NU, loadings_VD_sqrtN_inv, T