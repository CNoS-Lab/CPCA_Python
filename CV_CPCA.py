import numpy as np
from runCPCA import runCPCA
from runCPCA import findLoadingsScores
from G_CPCA_Analysis import G_CPCA_Analysis
from getPcaPredCorr import getPcaPredCorr
from getPcaPredCorr import getPcaPredCorr2
from splitHalfCrossValidCpca import splitHalfCrossValidCpca
from splitHalfCrossValidCpca import runCpcaStep1
from ipywidgets import IntProgress
from IPython.display import display
import time

def CV_CPCA(Z, G, params):
    G_CPCA_CV = splitHalfCrossValidCpca(Z, G, params)
    BH_PCA_test = getPcaPredCorr(G_CPCA_CV['G_CPCA_test'], params)
    BH_PCA_test['G_CPCA_CV'] = G_CPCA_CV
    BH_PCA_train = getPcaPredCorr2(G_CPCA_CV['G_CPCA_train'], params, BH_PCA_test['nComp'])
    BH_PCA_train['G_CPCA_CV'] = G_CPCA_CV
    ## perform permutation tests
    train_PCorr_shape=np.shape(BH_PCA_train['PCorr'])
    permutePCorrTrain = np.zeros((train_PCorr_shape[0],train_PCorr_shape[1],train_PCorr_shape[2],params['n_bootstrap']))
    permutePCorrTrain[:,:,:,:]=np.nan
    test_PCorr_shape=np.shape(BH_PCA_test['PCorr'])
    permutePCorrTest = np.zeros((test_PCorr_shape[0],test_PCorr_shape[1],test_PCorr_shape[2],params['n_bootstrap']))
    permutePCorrTest[:,:,:,:]=np.nan
    print('Please wait…') 
    max_count = params['n_bootstrap']-1
    f = IntProgress(min=0, max=max_count) # instantiate the bar
    display(f) # display the bar
    for ii in range(0,params['n_bootstrap']):
        G_temp = np.random.permutation(G) 
        Z_temp = Z
        G_CPCA_CV_temp = splitHalfCrossValidCpca(Z_temp, G_temp, params)
        BH_PCA_temp = getPcaPredCorr2(G_CPCA_CV_temp['G_CPCA_train'], params, BH_PCA_train['nComp'])
        permutePCorrTrain[:,:,:,ii] = BH_PCA_temp['PCorr']
        BH_PCA_temp = getPcaPredCorr2(G_CPCA_CV_temp['G_CPCA_test'], params, BH_PCA_test['nComp'])
        permutePCorrTest[:,:,:,ii] = BH_PCA_temp['PCorr']
        count=ii
        while count <= max_count:
            f.value += 1 # signal to increment the progress bar
            time.sleep(.01)
            break
        if ii+1==params['n_bootstrap']:
            print('Completed!')
                            
    #save('permutePCorr.mat','permutePCorrTest','permutePCorrTrain')
    Test_s=np.shape(permutePCorrTest)
    Train_s=np.shape(permutePCorrTrain)
    permutePCorr_mat=np.zeros((Train_s[0],Train_s[1]+Test_s[1],Train_s[2],Train_s[3]))
    for i in range(Train_s[3]):
        for j in range(Train_s[2]):
            for m in range(Train_s[1]+Test_s[1]):
                for n in range(Train_s[0]):
                    if m<Train_s[1]:
                        permutePCorr_mat[n,m,j,i]=permutePCorrTrain[n,m,j,i]
                    else:
                        permutePCorr_mat[n,m,j,i]=permutePCorrTest[n,m-Train_s[1],j,i]
    np.save('permutePCorr',permutePCorr_mat)

    BH_PCA_train['permutePCorrMean'] = np.nanmean(permutePCorrTrain,axis=2)
    BH_PCA_test['permutePCorrMean'] = np.nanmean(permutePCorrTest,axis=2)

    absPermutePCorrMean_list=list()
    for i in range(0,np.shape(BH_PCA_train['permutePCorrMean'])[2]):
        for j in range(0,np.shape(BH_PCA_train['permutePCorrMean'])[1]):
            for k in range(0,np.shape(BH_PCA_train['permutePCorrMean'])[0]):
                list.append(absPermutePCorrMean_list,abs(BH_PCA_train['permutePCorrMean'][k,j,i]))
    BH_PCA_train['predrel_cutoff'] = np.quantile(absPermutePCorrMean_list, 1-params['p_val'],method='nearest')
    absPermutePCorrMean_list=list()
    for i in range(0,np.shape(BH_PCA_test['permutePCorrMean'])[2]):
        for j in range(0,np.shape(BH_PCA_test['permutePCorrMean'])[1]):
            for k in range(0,np.shape(BH_PCA_test['permutePCorrMean'])[0]):
                list.append(absPermutePCorrMean_list,abs(BH_PCA_test['permutePCorrMean'][k,j,i]))
    BH_PCA_test['predrel_cutoff'] = np.quantile(absPermutePCorrMean_list, 1-params['p_val'],method='nearest')

    BH_PCA_train['meanPCorr'] = np.nanmean(BH_PCA_train['PCorr'],axis=2)
    BH_PCA_test['meanPCorr'] = np.nanmean(BH_PCA_test['PCorr'],axis=2)

    BH_PCA_train['RelPredLoadMask'] = abs(BH_PCA_train['meanPCorr']) >= BH_PCA_train['predrel_cutoff']
    BH_PCA_test['RelPredLoadMask'] = abs(BH_PCA_test['meanPCorr']) >= BH_PCA_test['predrel_cutoff']
    return BH_PCA_train, BH_PCA_test