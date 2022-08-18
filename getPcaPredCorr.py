import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import scipy.stats as stats
import math
import scipy.io as sio
from scipy.stats.stats import pearsonr 
from runCPCA import runCPCA
from runCPCA import findLoadingsScores

def estVar(X):
    [R,D,S] = la.svd(X,0)
    varOut = sum(D[:]**2)
    return varOut

def getVarianceTable(Z,S,E,Scores,Loadings):
    varStruct = {}
    varStruct['varZ'] = estVar(Z)
    varStruct['perVarZ'] = 100
    varStruct['varS'] = estVar(S)
    varStruct['perVarBH'] = varStruct['varS']/varStruct['varZ']*100
    varStruct['varE'] = estVar(E)
    varStruct['perVarE'] = varStruct['varE']/varStruct['varZ']*100
    varStruct['varScores'] = estVar(Scores@np.transpose(Loadings))
    varStruct['perVarScores'] = varStruct['varScores']/varStruct['varZ']*100
    varStruct['varScoresCompWise']={}
    varStruct['perVarScoresCompWise']={}
    varStruct['perVarScoresCompWiseBetween']={}
    for ii in range (np.shape(Scores)[1]):
        S=np.array(Scores[:,ii])
        L=np.array(Loadings[:,ii])
        S=S.reshape(len(S),1)
        L=L.reshape(len(L),1)
        varStruct['varScoresCompWise'][str(ii)] = estVar(S*np.transpose(L))
        varStruct['perVarScoresCompWise'][str(ii)] = varStruct['varScoresCompWise'][str(ii)]/varStruct['varZ']*100
        varStruct['perVarScoresCompWiseBetween'][str(ii)] = varStruct['varScoresCompWise'][str(ii)]/varStruct['varS']*100
    return varStruct

def getPcaPredCorr(G_CPCA_CV, params):
    n_ite=2*params['nIter']
    BH_big=np.zeros(((n_ite*np.shape(G_CPCA_CV['0']['S'])[0]),np.shape(G_CPCA_CV['0']['S'])[1]))
    for ii in range(0,n_ite):
        BH_big[np.shape(G_CPCA_CV['0']['S'])[0]*ii:np.shape(G_CPCA_CV['0']['S'])[0]*(ii+1),:] = G_CPCA_CV[str(ii)]['S']#BH_big = np.concatenate(S)
    G_big_cell={}
    for ii in range(0,n_ite):
        G_big_cell[str(ii)]=G_CPCA_CV[str(ii)]['G']#G_big_cell = {G_CPCA_CV.G};
        G_big_cell[str(ii)]=np.vstack(G_big_cell[str(ii)])
    n_temp=la.matrix_rank(BH_big)
    BH_PCA_temp = runCPCA(BH_big,1,1,n_temp,params['varimaxFlag'])
    D = BH_PCA_temp['DGH']
    D = np.power(np.diag(D),2)
    D = D/sum(D)*100
    k = len(D);
    x=list()
    y=list()
    for i in range(0,k):
        list.append(x,i+1)
    for i in range(0,k):
        list.append(y,D[i])
    s = plt.plot(x,y,'o-',markerfacecolor='none')
    plt.xlabel('No. of Components')
    plt.xticks(np.arange(0,k+1,step=5))
    plt.ylabel('Percentage of variance')
    plt.title('Scree Plot')
    plt.show(s)
    nComp=int(input("Enter the number of components:"))
        
    BH_PCA = runCPCA(BH_big,1,1,nComp,params['varimaxFlag'])
    BH_PCA['nComp'] = nComp
    nVals = n_ite
    BH_PCA['PCorr']=np.zeros((np.shape(G_CPCA_CV[str(ii)]['G'])[1], nComp, nVals))
    BH_PCA['PCorr'][:,:,:]=np.nan
    first_idx = 0
    BH_PCA['scores_NU_GH_cell']={}
    for ii in range(0,nVals):
        last_idx = first_idx + np.shape(G_big_cell[str(ii)])[0]
        BH_PCA['scores_NU_GH_cell'][str(ii)] = BH_PCA['scores_NU_GH'][first_idx:last_idx,:]
        #BH_PCA['PCorr'][:,:,ii] = corr(G_big_cell[str(ii)], BH_PCA['scores_NU_GH_cell'][str(ii)]);
        for i in range(0,(np.shape(G_CPCA_CV[str(ii)]['G'])[1])):
            for j in range(0,nComp):
                BH_PCA['PCorr'][i,j,ii]=(pearsonr(G_big_cell[str(ii)][:,i],BH_PCA['scores_NU_GH_cell'][str(ii)][:,j]))[0]
        first_idx = last_idx
        BH_PCA['varianceTable']={}
        BH_PCA['varianceTable'][str(ii)] = getVarianceTable(G_CPCA_CV[str(ii)]['Z'],G_CPCA_CV[str(ii)]['S'],G_CPCA_CV[str(ii)]['E'],BH_PCA['scores_NU_GH_cell'][str(ii)],BH_PCA['loadings_VD_sqrtN_GH'])
    BH_PCA['subIdx_big']=np.zeros(((n_ite*np.shape(G_CPCA_CV['0']['subIdx'])[0]),))
    for ii in range(0,n_ite):
        BH_PCA['subIdx_big'][np.shape(G_CPCA_CV[str(ii)]['subIdx'])[0]*ii:np.shape(G_CPCA_CV[str(ii)]['subIdx'])[0]*(ii+1),] = G_CPCA_CV[str(ii)]['subIdx']#BH_PCA.subIdx_big = cat(1,G_CPCA_CV.subIdx)
    subIdx = sorted(np.unique(BH_PCA['subIdx_big']))
    subIdx.reverse
    BH_PCA['meanScores'] = np.zeros((np.size(subIdx),nComp))
    BH_PCA['meanScores'][:,:] = np.nan
    for sub in subIdx:
        mean_temp=list()
        for i in range(0,np.shape(BH_PCA['subIdx_big'])[0]):
            if BH_PCA['subIdx_big'][i,] == sub:
                list.append(mean_temp,BH_PCA['scores_NU_GH'][i,:])
        BH_PCA['meanScores'][int(sub)-1,:] = np.mean(mean_temp,axis=0)
    #BH_PCA.meanScores_PCorr = corr(params.G_orig, BH_PCA.meanScores);
    BH_PCA['meanScores_PCorr'] = np.zeros((np.shape(params['G_orig'])[1],np.shape(BH_PCA['meanScores'])[1]))
    for i in range(0,np.shape(params['G_orig'])[1]):
        for j in range(0,np.shape(BH_PCA['meanScores'])[1]):
            BH_PCA['meanScores_PCorr'][i,j]=(pearsonr(params['G_orig'][:,i],BH_PCA['meanScores'][:,j]))[0]
    BH_PCA['G_big_cell'] = G_big_cell
    return BH_PCA

def getPcaPredCorr2(G_CPCA_CV, params,nComp):
    n_ite=2*params['nIter']
    BH_big=np.zeros(((n_ite*np.shape(G_CPCA_CV['0']['S'])[0]),np.shape(G_CPCA_CV['0']['S'])[1]))
    for ii in range(0,n_ite):
        BH_big[np.shape(G_CPCA_CV['0']['S'])[0]*ii:np.shape(G_CPCA_CV['0']['S'])[0]*(ii+1),:] = G_CPCA_CV[str(ii)]['S']#BH_big = np.concatenate(S)
    G_big_cell={}
    for ii in range(0,n_ite):
        G_big_cell[str(ii)]=G_CPCA_CV[str(ii)]['G']#G_big_cell = {G_CPCA_CV.G};
        G_big_cell[str(ii)]=np.vstack(G_big_cell[str(ii)])
        
    BH_PCA = runCPCA(BH_big,1,1,nComp,params['varimaxFlag'])
    BH_PCA['nComp'] = nComp
    nVals = n_ite
    BH_PCA['PCorr']=np.zeros((np.shape(G_CPCA_CV[str(ii)]['G'])[1], nComp, nVals))
    BH_PCA['PCorr'][:,:,:]=np.nan
    first_idx = 0
    BH_PCA['scores_NU_GH_cell']={}
    for ii in range(0,nVals):
        last_idx = first_idx + np.shape(G_big_cell[str(ii)])[0]
        BH_PCA['scores_NU_GH_cell'][str(ii)] = BH_PCA['scores_NU_GH'][first_idx:last_idx,:]
        #BH_PCA['PCorr'][:,:,ii] = corr(G_big_cell[str(ii)], BH_PCA['scores_NU_GH_cell'][str(ii)]);
        for i in range(0,(np.shape(G_CPCA_CV[str(ii)]['G'])[1])):
            for j in range(0,nComp):
                BH_PCA['PCorr'][i,j,ii]=(pearsonr(G_big_cell[str(ii)][:,i],BH_PCA['scores_NU_GH_cell'][str(ii)][:,j]))[0]
        first_idx = last_idx
        BH_PCA['varianceTable']={}
        BH_PCA['varianceTable'][str(ii)] = getVarianceTable(G_CPCA_CV[str(ii)]['Z'],G_CPCA_CV[str(ii)]['S'],G_CPCA_CV[str(ii)]['E'],BH_PCA['scores_NU_GH_cell'][str(ii)],BH_PCA['loadings_VD_sqrtN_GH'])
    BH_PCA['subIdx_big']=np.zeros(((n_ite*np.shape(G_CPCA_CV['0']['subIdx'])[0]),))
    for ii in range(0,n_ite):
        BH_PCA['subIdx_big'][np.shape(G_CPCA_CV[str(ii)]['subIdx'])[0]*ii:np.shape(G_CPCA_CV[str(ii)]['subIdx'])[0]*(ii+1),] = G_CPCA_CV[str(ii)]['subIdx']#BH_PCA.subIdx_big = cat(1,G_CPCA_CV.subIdx)
    subIdx = sorted(np.unique(BH_PCA['subIdx_big']))
    subIdx.reverse
    BH_PCA['meanScores'] = np.zeros((np.size(subIdx),nComp))
    BH_PCA['meanScores'][:,:] = np.nan
    for sub in subIdx:
        mean_temp=list()
        for i in range(0,np.shape(BH_PCA['subIdx_big'])[0]):
            if BH_PCA['subIdx_big'][i,] == sub:
                list.append(mean_temp,BH_PCA['scores_NU_GH'][i,:])
        BH_PCA['meanScores'][int(sub)-1,:] = np.mean(mean_temp,axis=0)
    #BH_PCA.meanScores_PCorr = corr(params.G_orig, BH_PCA.meanScores);
    BH_PCA['meanScores_PCorr'] = np.zeros((np.shape(params['G_orig'])[1],np.shape(BH_PCA['meanScores'])[1]))
    for i in range(0,np.shape(params['G_orig'])[1]):
        for j in range(0,np.shape(BH_PCA['meanScores'])[1]):
            BH_PCA['meanScores_PCorr'][i,j]=(pearsonr(params['G_orig'][:,i],BH_PCA['meanScores'][:,j]))[0]
    BH_PCA['G_big_cell'] = G_big_cell
    return BH_PCA