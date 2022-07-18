import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import scipy.stats as stats
import math
import scipy.io
from scipy.stats.stats import pearsonr 
from runCPCA import runCPCA
from runCPCA import findLoadingsScores

def G_CPCA_Analysis(Z, G, n, varmaxFlag):
    
    varmaxFlag=False

    G_CPCA={}
    # Code to run CPCA using G matrix

    Z = (Z - np.nanmean(Z,axis=0))/np.nanstd(Z,axis=0,ddof=1) # Z-score the data 
    Z = (np.nan_to_num(Z, nan=0.0))

    if np.all(G==[]):
        G_CPCA = runCPCA(Z,1,1,la.matrix_rank(Z),varmaxFlag)  # Uncontrained PCA
        return G_CPCA

    if ~np.all(n==[]):
        G_CPCA = runCPCA(Z,G,1,n,varmaxFlag) # PCA model
    else:
        G_CPCA = runCPCA(Z,G,1,min(la.matrix_rank(Z),la.matrix_rank(G)),varmaxFlag)  # CPCA model
    
    # use two 'for' loop and pearsonr to realize the function 'corr()' in MATLAB
    rG,cG=np.shape(G)
    rs,cs=np.shape(G_CPCA['scores_NU_GH'])
    G_CPCA['PCorr']=np.zeros((cG,cs))
    G_CPCA['PCorr_pval']=np.zeros((cG,cs))
    for i in range(0,cG):
        for j in range(0,cs):
            G_CPCA['PCorr'][i,j], G_CPCA['PCorr_pval'][i,j]=pearsonr(G[:,i],G_CPCA['scores_NU_GH'][:,j])  # Predictor correlations
            
# Save the Z and G matrices in the output struct

    G_CPCA['G'] = G
    G_CPCA['Z'] = Z
    G_CPCA['n'] = n

    G_CPCA['varG'] = []
    G_CPCA['predictStruct'] = []

    return G_CPCA