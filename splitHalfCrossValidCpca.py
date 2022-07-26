import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import scipy.stats as stats
import math
import scipy.io

#runCPCAstep1
#RunCPCA 1st cell
####  may need to turns NaN into nan, which could be identified by matlab
def runCpcaStep1(Z, G, H):      
# X - rows are observations (Points), columns are features (variables)
# X should be zscored along the columns
# G - Predictor variable matrix along 1st dim -rows
# H - Predictor variable matrix along 2nd dim - cols
# n - Select top n components
    OutStruct = {}
    if  np.size(G) != 1 and H == 1:  ###one of GH are 1
        C = la.pinv(np.transpose(G)@G)@np.transpose(G)@Z # Linear regression
        S = G@C
        E = Z - S
        OutStruct['C'] = C
        OutStruct['B'] = np.nan
    elif G == 1 and np.size(H) != 1:
        B = Z@H@la.pinv(np.transpose(H)@H) # Linear regression BH + E
        S = B@np.transpose(H)
        E = Z - S
        OutStruct['B'] = B
        OutStruct['C'] = np.nan
    elif G == 1 and H == 1:
        S = Z
        E = Z - S
    else:
        return
    OutStruct['Z'] = Z
    OutStruct['G'] = G
    OutStruct['H'] = H
    OutStruct['E'] = E
    OutStruct['S'] = S
    OutStruct['subIdx'] = np.nan
    return OutStruct

from random import shuffle
from random import seed
def crossvalind(size):
    sequence = [i for i in range(size)]
    c=np.zeros((1,size)).T
    shuffle(sequence)
    for i in range(size):
        if(sequence[i]<int(size/2)):
            c[i]=1
        else:
            c[i]=2
    return c

def splitHalfCrossValidCpca(Z, G, params):
    nIter = params["nIter"]
    size=np.shape(Z)[0]
    cv_idx=np.zeros((size,nIter))
    G_CPCA_train_1 = {}
    G_CPCA_train_2 = {}
    G_CPCA_test_1 = {}
    G_CPCA_test_2 = {}
    for ii in range(0,nIter):
        c=crossvalind(size)
        for j in range(0,size):
            cv_idx[j,ii] = c[j]
    Z = (Z - np.nanmean(Z,axis=0))/np.nanstd(Z,axis=0,ddof=1) # Z-score the data 
    #Z = (np.nan_to_num(Z, nan=0.0))
    Z_temp = list()
    G_temp = list()
    for ii in range(0,nIter):
        for i in range (0,np.shape(Z)[0]):
            if cv_idx[i,ii] == 1:
                list.append(Z_temp,Z[i,:])
        for i in range (0,np.shape(G)[0]):
            if cv_idx[i,ii] == 1:
                list.append(G_temp,G[i,:])
        G_CPCA_train_1[str(ii)] = runCpcaStep1(Z_temp, G_temp, 1)
        Z_temp = []
        G_temp = []
        G_CPCA_train_1[str(ii)]['subIdx'] = np.where(cv_idx[:,ii]==1)[0]
        for i in range (0,np.shape(Z)[0]):
            if cv_idx[i,ii] == 2:
                list.append(Z_temp,Z[i,:])
        G_CPCA_test_1[str(ii)] = runCpcaStep1(Z_temp, 1, np.transpose(G_CPCA_train_1[str(ii)]['C']))
        Z_temp = []
        for i in range (0,np.shape(G)[0]):
            if cv_idx[i,ii] == 2:
                list.append(G_temp,G[i,:])
        G_CPCA_test_1[str(ii)]['G'] = G_temp
        G_temp = []
        G_CPCA_test_1[str(ii)]['subIdx'] = np.where(cv_idx[:,ii]==2)[0]
        for i in range (0,np.shape(Z)[0]):
            if cv_idx[i,ii] == 2:
                list.append(Z_temp,Z[i,:])
        for i in range (0,np.shape(G)[0]):
            if cv_idx[i,ii] == 2:
                list.append(G_temp,G[i,:])
        G_CPCA_train_2[str(ii)] = runCpcaStep1(Z_temp, G_temp, 1)
        Z_temp = []
        G_temp = []
        G_CPCA_train_2[str(ii)]['subIdx'] = np.where(cv_idx[:,ii]==2)[0]
        for i in range (0,np.shape(Z)[0]):
            if cv_idx[i,ii] == 1:
                list.append(Z_temp,Z[i,:])
        G_CPCA_test_2[str(ii)] = runCpcaStep1(Z_temp, 1, np.transpose(G_CPCA_train_2[str(ii)]['C']))
        Z_temp = []
        for i in range (0,np.shape(G)[0]):
            if cv_idx[i,ii] == 1:
                list.append(G_temp,G[i,:])
        G_CPCA_test_2[str(ii)]['G'] = G_temp
        G_temp = []
        G_CPCA_test_2[str(ii)]['subIdx'] = np.where(cv_idx[:,ii]==1)[0]  
    
    outStruct={}
    outStruct['G_CPCA_train']={}
    outStruct['G_CPCA_test']={}
    for ii in range(0,2*nIter):
        if ii<nIter:
            outStruct['G_CPCA_train'][str(ii)]=G_CPCA_train_1[str(ii)]
        else:
            outStruct['G_CPCA_train'][str(ii)]=G_CPCA_train_2[str(ii-nIter)]
    for ii in range(0,2*nIter):
        if ii<nIter:
            outStruct['G_CPCA_test'][str(ii)]=G_CPCA_test_1[str(ii)]
        else:
            outStruct['G_CPCA_test'][str(ii)]=G_CPCA_test_2[str(ii-nIter)] 
    outStruct['cv_idx'] = cv_idx
    outStruct['params'] = params
    outStruct['Z'] = Z
    outStruct['G'] = G
    return outStruct