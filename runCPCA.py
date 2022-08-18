import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import scipy.stats as stats
import math
import scipy.io
#RunCPCA 1st cell
####  may need to turns NaN into nan, which could be identified by matlab
def varim_rotation(lam, eps=1e-6, itermax=100):
    gamma = 1.0
    nrow, ncol = lam.shape
    R = np.eye(ncol)
    var = 0
    for i in range(itermax):
        lam_rot = np.dot(lam, R)
        tmp = np.diag(np.sum(lam_rot ** 2, axis=0)) / nrow * gamma
        u, s, v = np.linalg.svd(np.dot(lam.T, lam_rot ** 3 - np.dot(lam_rot, tmp)))
        R = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < var * (1 + eps):
            break
        var = var_new
    return lam_rot, R

def varimkn_beh(INPMAT):
    SUMI=sum((INPMAT**2).T)
    SUMINP=np.sqrt(SUMI)

    m, n = INPMAT.shape
    KAISNOR=np.zeros((m, n))
    AFIN=np.zeros((m, n))
    for count1 in range(m):
        for count2 in range(n):
            KAISNOR[count1,count2]=INPMAT[count1,count2]/SUMINP[count1]
    # print(KAISNOR)
    VARK, T = varim_rotation(KAISNOR, eps=1e-6, itermax=100)

    for count1 in range(m):
        for count2 in range(n):
            AFIN[count1,count2]=VARK[count1,count2]*SUMINP[count1]
    return AFIN, T

def runCPCA(Z, G, H, n, varmaxFlag):      
# X - rows are observations (Points), columns are features (variables)
# X should be zscored along the columns
# G - Predictor variable matrix along 1st dim -rows
# H - Predictor variable matrix along 2nd dim - cols
# n - Select top n components
    # varmaxFlag=False
    N,NNN = np.shape(Z)    ########deal with size, no of rows in Z  
    if  np.size(G) != 1 and H == 1:  ###one of GH are 1
        #C = la.pinv(G.T@G)@G.T@Z # Linear regression
        slr = LinearRegression(fit_intercept=False)
        slr.fit(G, Z)
        y_pred = slr.predict(G)
        C = np.transpose(slr.coef_)
        S = G@C
        E = Z - S
    elif G == 1 and np.size(H) != 1:
        #B = Z@H@la.pinv(np.transpose(H)@H) # Linear regression
        slr = LinearRegression(fit_intercept=False)
        Z_temp=np.transpose(Z)
        slr.fit(H, Z_temp)
        y_pred = slr.predict(H)
        B = slr.coef_
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
    [OutStruct['loadings_VD_sqrtN_GH'], OutStruct['scores_NU_GH'], OutStruct['loadings_VD_sqrtN_GH_inv'], OutStruct['T']] = findLoadingsScores(OutStruct['UGH'], OutStruct['DGH'], OutStruct['VGH'], N, n, varmaxFlag)
    
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
    
    if varmaxFlag == True and n > 1:
        loadings_VD_sqrtN, T= varimkn_beh(loadings_VD_sqrtN)
        loadings_VD_sqrtN_inv = loadings_VD_sqrtN_inv@T
        scores_NU = scores_NU@T
    
    return loadings_VD_sqrtN, scores_NU, loadings_VD_sqrtN_inv, T