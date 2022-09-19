import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import scipy.stats as stats
import math
import scipy.io
from scipy.stats.stats import pearsonr 
from splitHalfCrossValidCpca import runCpcaStep1

def SelectCompLoadingsBig(BH_PCA, params):

    Res={}
    Z_dim = np.shape(BH_PCA['Z'])[1]

    for ii in range(Z_dim):
        Res[str(ii)]={}
        Rej_feat = np.array([ii])
        All_feat = np.arange(Z_dim)
        Sel_feat = np.setdiff1d(All_feat, Rej_feat)
        
        N = np.shape(BH_PCA['Z'])[0]
        G_CPCA = runCpcaStep1(BH_PCA['Z'][:,Sel_feat], BH_PCA['Z'][:,Rej_feat], 1)
        Res[str(ii)]['ResidueS'] = G_CPCA['E']
        Res[str(ii)]['ResidueScores'] = math.sqrt(N-1) * Res[str(ii)]['ResidueS'] @ BH_PCA['VGH'][Sel_feat,:] @ la.pinv(BH_PCA['DGH'])
        
        if params['varimaxFlag']:
            Res[str(ii)]['ResidueScoresRot'] = Res[str(ii)]['ResidueScores'][:,0:BH_PCA['nComp']]@BH_PCA['T']
        else:
            Res[str(ii)]['ResidueScoresRot'] = Res[str(ii)]['ResidueScores'][:,0:BH_PCA['nComp']]
        # Break-up the big component score matrix into individual chunks
        # and estimate predictor correlations

        nVals = len(BH_PCA['G_big_cell'])
        Res[str(ii)]['ResiduePCorr'] = np.zeros(np.shape(BH_PCA['PCorr']))
        Res[str(ii)]['ResiduePCorr'][:,:,:]=np.nan
        first_idx = 0
        Res[str(ii)]['ResidueScoresRot_cell']={}
        for jj in range(nVals):
            last_idx = first_idx + np.shape(BH_PCA['G_big_cell'][str(jj)])[0]
            Res[str(ii)]['ResidueScoresRot_cell'][str(jj)] = Res[str(ii)]['ResidueScoresRot'][first_idx:last_idx,:]
            for i in range(np.shape(BH_PCA['G_big_cell'][str(jj)])[1]):
                for j in range(np.shape(Res[str(ii)]['ResidueScoresRot_cell'][str(jj)])[1]):
                    Res[str(ii)]['ResiduePCorr'][i,j,jj]=(pearsonr(BH_PCA['G_big_cell'][str(jj)][:,i],Res[str(ii)]['ResidueScoresRot_cell'][str(jj)][:,j]))[0]
            first_idx = last_idx

        if last_idx != np.shape(BH_PCA['scores_NU_GH'])[0]:
            print('Error in break-up of comp. score matrix.')
            return
    
        Res[str(ii)]['meanPCorr'] = np.nanmean(Res[str(ii)]['ResiduePCorr'],axis=2)
        Res[str(ii)]['SumPredLoad'] = sum(abs(Res[str(ii)]['meanPCorr']))
        Res[str(ii)]['SumPredLoadRel'] = sum(abs(Res[str(ii)]['meanPCorr']*BH_PCA['RelPredLoadMask']))
        Res[str(ii)]['Rej_feat'] = Rej_feat
        Res[str(ii)]['Sel_feat'] = Sel_feat

    return Res