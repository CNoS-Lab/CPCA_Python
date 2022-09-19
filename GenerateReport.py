import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SelectCompLoadingsBig import SelectCompLoadingsBig
import plotly.graph_objects as go

def PlotLoadings(result,Label,Dataset,Loadings):
    if Loadings=='Component':
        temp = result['loadings_VD_sqrtN_GH']
        y_ticks = Label['Z_label']
    elif Loadings=='Predictor':
        temp = result['meanPCorr']
        y_ticks = Label['G_label']
    x_L,y_L = np.shape(temp)
    x_ticks=list()
    for i in range(0,y_L):
        list.append(x_ticks,"C%d"%(i+1))
    plt.figure(dpi=350,figsize=(y_L*4,x_L))
    h=sns.heatmap(temp,annot=True,cmap='RdBu_r', center=0,xticklabels=x_ticks,yticklabels=y_ticks,linewidths=0.2,linecolor='black',square=True,cbar=False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12,rotation=0)
    #plt.annotate()#
    h.set_title(Dataset+' data - '+Loadings+' Loadings')
    plt.show()
    
def PredLThld(result,params,Dataset):
    print(Dataset+' data - Predictor loading threshold - p <=',params['p_val'])
    print('Estimated Predictor loading threshold =',result['predrel_cutoff'],'(at p<=',params['p_val'],')')
    predRel=list()
    ppm_shape=np.shape(result['permutePCorrMean'])
    for i in range(ppm_shape[2]):
        for j in range(ppm_shape[1]):
            for k in range(ppm_shape[0]):
                list.append(predRel,abs(result['permutePCorrMean'][k,j,i]))
    his=plt.hist(predRel,bins=50,color='lightskyblue',edgecolor='k')
    plt.vlines(np.quantile(predRel, 1-0.05,method='nearest'),ymin=0,ymax=max(his[0]),colors='r',linestyles='dotted',label='p=0.05')
    plt.vlines(np.quantile(predRel, 1-0.01,method='nearest'),ymin=0,ymax=max(his[0]),colors='m',linestyles='dotted',label='p=0.01')
    plt.vlines(np.quantile(predRel, 1-0.005,method='nearest'),ymin=0,ymax=max(his[0]),colors='g',linestyles='dotted',label='p=0.005')
    plt.vlines(np.quantile(predRel, 1-0.001,method='nearest'),ymin=0,ymax=max(his[0]),colors='b',linestyles='dotted',label='p=0.001')
    plt.legend()
    plt.title(Dataset+' data - Predictor loading threshold')
    plt.show()
    
def LOO_domCL(result,params,Label):
    Z_label=Label['Z_label']
    result['Res']=SelectCompLoadingsBig(result, params)
    comp_rel=np.where(sum(result['RelPredLoadMask']))
    sumPredLoad=result['Res']['0']['SumPredLoad']
    for ii in range(1,len(result['Res'])):
        sumPredLoad=np.row_stack((sumPredLoad,result['Res'][str(ii)]['SumPredLoad']))
    sumPredLoadRel=result['Res']['0']['SumPredLoadRel']
    for ii in range(1,len(result['Res'])):
        sumPredLoadRel=np.row_stack((sumPredLoadRel,result['Res'][str(ii)]['SumPredLoadRel']))
    sumPredLoadSort=np.zeros(np.shape(sumPredLoad))
    sumPredLoadSortIdx=np.zeros(np.shape(sumPredLoad))
    for ii in range(np.shape(sumPredLoad)[1]):
        sumPredLoadSort[:,ii] = sorted(sumPredLoad[:,ii])
        sumPredLoadSortIdx[:,ii] = sorted(range(len(sumPredLoad[:,ii])), key = lambda x:sumPredLoad[x,ii])
    sumPredLoadRelSort=np.zeros(np.shape(sumPredLoadRel))
    sumPredLoadRelSortIdx=np.zeros(np.shape(sumPredLoadRel))
    for ii in range(np.shape(sumPredLoadRel)[1]):
        sumPredLoadRelSort[:,ii] = sorted(sumPredLoadRel[:,ii])
        sumPredLoadRelSortIdx[:,ii] = sorted(range(len(sumPredLoadRel[:,ii])), key = lambda x:sumPredLoadRel[x,ii])
    for ii in comp_rel[0]:
        plt.figure()
        s = plt.plot(np.arange(np.shape(sumPredLoadRelSort)[0]),sumPredLoadRelSort[:,ii],'o-')
        xticks=list()
        for i in range(np.shape(sumPredLoadRelSortIdx)[0]):
            list.append(xticks,str(int(sumPredLoadRelSortIdx[i,ii]+1)))
        plt.xlabel('Variable dominance rank #')
        plt.xticks(np.arange(np.shape(sumPredLoadRelSort)[0]),xticks)
        plt.ylabel('Sum of All Pred. Load.')
        plt.title('Comp %d - Sum of reliable Pred. Load.'%(ii+1))
        plt.show()
        for i in range(len(xticks)):
            print(xticks[i], Z_label[int(xticks[i])-1])
            
def sankey_diagram_2(result,params,Label):
    Z_label=Label['Z_label']
    G_label=Label['G_label']
    result['Res']=SelectCompLoadingsBig(result, params)
    sumPredLoadRel=result['Res']['0']['SumPredLoadRel']
    for ii in range(1,len(result['Res'])):
        sumPredLoadRel=np.row_stack((sumPredLoadRel,result['Res'][str(ii)]['SumPredLoadRel']))
    sumPredLoadRelSort=np.zeros(np.shape(sumPredLoadRel))
    sumPredLoadRelSortIdx=np.zeros(np.shape(sumPredLoadRel))
    for ii in range(np.shape(sumPredLoadRel)[1]):
        sumPredLoadRelSort[:,ii] = sorted(sumPredLoadRel[:,ii])
        sumPredLoadRelSortIdx[:,ii] = sorted(range(len(sumPredLoadRel[:,ii])), key = lambda x:sumPredLoadRel[x,ii])
    comp_rel=np.where(sum(result['RelPredLoadMask']))[0]
    n_comp_nodes=len(comp_rel)
    n_vdom=np.zeros((result['nComp'],1))
    n_vdom=n_vdom.astype(np.int32)
    sPred=list()
    sPredVal=list()
    pred_yticks=list()
    pred_nodes=list()
    comp_yticks=list()
    comp_nodes=list()
    var_yticks=list()
    var_nodes=list()
    vDomVal=list()
    sPredVal=list()
    sPred_color=list()
    sPred_comp=list()
    vDom_color=list()
    vDom_comp=list()
    comp_color_pred=list()
    comp_color_comp=list()
    colors_pool=['#ffdc73','#568cb4','#ad85d2','gold','darkgray']
    pl_label_a=list()
    c_label_a=list()
    cl_label_a=list()
    comp_label_flag=-1
    for ii in comp_rel:
        comp_label_flag=comp_label_flag+1
        n_vdom[ii,0]=input('Number of Variables for Component %d:'%(ii+1))
        vdom_indx=sumPredLoadRelSortIdx[0:n_vdom[ii,0],ii]
        vDom=list()
        sPred=list()
        list.append(c_label_a,'auto')
        for i in range(np.shape(result['PCorr'])[0]):
            if(result['RelPredLoadMask'][i,ii]==True):
                list.append(sPred,int(i+1))
                list.append(pred_yticks,int(i+1))
                list.append(sPredVal,abs(result['meanPCorr'][i,ii]))
                list.append(sPred_comp,comp_label_flag)
                # if result['meanPCorr'][i,ii]>0:
                #     list.append(sPred_color,'lightcoral')
                # else:
                #     list.append(sPred_color,'royalblue')
                list.append(comp_color_pred,colors_pool[comp_label_flag])
                list.append(pl_label_a,'left')
        for i in vdom_indx:
            list.append(vDom,int(sumPredLoadRelSortIdx[int(i),ii]+1))
            list.append(vDomVal,abs(result['loadings_VD_sqrtN_GH'][int(i),ii]))
            list.append(var_yticks,int(int(i+1)))
            list.append(vDom_comp,comp_label_flag)
            # if result['meanPCorr'][i,ii]>0:
            #     list.append(vDom_color,'lightcoral')
            # else:
            #     list.append(vDom_color,'royalblue')
            list.append(comp_color_comp,colors_pool[comp_label_flag])
            list.append(cl_label_a,'right')
        if(~np.any(sPred)):
            list.append(sPred,np.nan)
        if(~np.any(vDom)):
            list.append(vDom,np.nan)
    nodes_label_align=pl_label_a+c_label_a+cl_label_a
    pred_y_nodes=list(set(pred_yticks))
    var_y_nodes=list(set(var_yticks))
    comp_yticks=np.linspace(1,np.size(comp_rel),num=np.size(comp_rel))
    for i in range(len(comp_rel)):
        list.append(comp_nodes,"C%d"%(comp_rel[i]+1))
    for i in range(len(pred_y_nodes)):
        list.append(pred_nodes,G_label[pred_y_nodes[i]-1])#pred_name[i]
    for i in range(len(var_y_nodes)):
        list.append(var_nodes,Z_label[var_y_nodes[i]-1])#pred_name[i]
    nodes_label=comp_nodes+pred_nodes+var_nodes
    comp_color=comp_color_pred+comp_color_comp
    source_predL=np.zeros(np.shape(pred_yticks))
    source_compL=np.zeros(np.shape(var_yticks))
    for ii in range(len(pred_yticks)):
        source_predL[ii]=pred_y_nodes.index(pred_yticks[ii])+n_comp_nodes
    for ii in range(len(var_yticks)):
        source_compL[ii]=var_y_nodes.index(var_yticks[ii])+n_comp_nodes+len(pred_nodes)
    link_source=np.hstack((source_predL,vDom_comp))
    link_value=np.hstack((sPredVal,vDomVal))
    link_color=comp_color#sPred_color+vDom_color
    link_target=np.hstack((sPred_comp,source_compL))
    fig = go.Figure(data=[go.Sankey(
        arrangement = "perpendicular",
        node = dict(
          pad = 20,
          thickness = 4,
          line = dict(color = "gray", width = 0.2),
          #label = nodes_label,
          color = "gray"
          ),
        link = dict(
          source = link_source, # indices correspond to labels, eg A1, A2, A1, B1, ...
          target = link_target,
          value = link_value,
          color = link_color
      ),
        hoverlabel=dict(
              align=nodes_label_align
      )
    )])

    fig.update_layout(title_text="Connection Diagram", font_size=10)
    fig.show()
    
def sankey_diagram(result,params,Label):
    Z_label=Label['Z_label']
    G_label=Label['G_label']
    result['Res']=SelectCompLoadingsBig(result, params)
    sumPredLoadRel=result['Res']['0']['SumPredLoadRel']
    for ii in range(1,len(result['Res'])):
        sumPredLoadRel=np.row_stack((sumPredLoadRel,result['Res'][str(ii)]['SumPredLoadRel']))
    sumPredLoadRelSort=np.zeros(np.shape(sumPredLoadRel))
    sumPredLoadRelSortIdx=np.zeros(np.shape(sumPredLoadRel))
    for ii in range(np.shape(sumPredLoadRel)[1]):
        sumPredLoadRelSort[:,ii] = sorted(sumPredLoadRel[:,ii])
        sumPredLoadRelSortIdx[:,ii] = sorted(range(len(sumPredLoadRel[:,ii])), key = lambda x:sumPredLoadRel[x,ii])
    comp_rel=np.where(sum(result['RelPredLoadMask']))[0]
    n_comp_nodes=len(comp_rel)
    n_vdom=np.zeros((result['nComp'],1))
    n_vdom=n_vdom.astype(np.int32)
    sPred=list()
    sPredVal=list()
    pred_yticks=list()
    pred_nodes=list()
    comp_yticks=list()
    comp_nodes=list()
    var_yticks=list()
    var_nodes=list()
    vDomVal=list()
    sPredVal=list()
    sPred_color=list()
    sPred_comp=list()
    vDom_color=list()
    vDom_comp=list()
    comp_color_pred=list()
    comp_color_comp=list()
    colors_pool=['#ffdc73','#568cb4','#ad85d2','gold','darkgray']
    pl_label_a=list()
    c_label_a=list()
    cl_label_a=list()
    comp_label_flag=-1
    for ii in comp_rel:
        comp_label_flag=comp_label_flag+1
        n_vdom[ii,0]=input('Number of Variables for Component %d:'%(ii+1))
        vdom_indx=sumPredLoadRelSortIdx[0:n_vdom[ii,0],ii]
        vDom=list()
        sPred=list()
        list.append(c_label_a,'auto')
        for i in range(np.shape(result['PCorr'])[0]):
            if(result['RelPredLoadMask'][i,ii]==True):
                list.append(sPred,int(i+1))
                list.append(pred_yticks,int(i+1))
                list.append(sPredVal,abs(result['meanPCorr'][i,ii]))
                list.append(sPred_comp,comp_label_flag)
                # if result['meanPCorr'][i,ii]>0:
                #     list.append(sPred_color,'lightcoral')
                # else:
                #     list.append(sPred_color,'royalblue')
                list.append(comp_color_pred,colors_pool[comp_label_flag])
                list.append(pl_label_a,'left')
        for i in vdom_indx:
            list.append(vDom,int(sumPredLoadRelSortIdx[int(i),ii]+1))
            list.append(vDomVal,abs(result['loadings_VD_sqrtN_GH'][int(i),ii]))
            list.append(var_yticks,int(i+1))
            list.append(vDom_comp,comp_label_flag)
            # if result['meanPCorr'][i,ii]>0:
            #     list.append(vDom_color,'lightcoral')
            # else:
            #     list.append(vDom_color,'royalblue')
            list.append(comp_color_comp,colors_pool[comp_label_flag])
            list.append(cl_label_a,'right')
        if(~np.any(sPred)):
            list.append(sPred,np.nan)
        if(~np.any(vDom)):
            list.append(vDom,np.nan)
    nodes_label_align=pl_label_a+c_label_a+cl_label_a
    pred_y_nodes=list(set(pred_yticks))
    var_y_nodes=list(set(var_yticks))
    comp_yticks=np.linspace(1,np.size(comp_rel),num=np.size(comp_rel))
    for i in range(len(comp_rel)):
        list.append(comp_nodes,"C%d"%(comp_rel[i]+1))
    for i in range(len(pred_y_nodes)):
        list.append(pred_nodes,G_label[pred_y_nodes[i]-1])#pred_name[i]
    for i in range(len(var_y_nodes)):
        list.append(var_nodes,Z_label[var_y_nodes[i]-1])#pred_name[i]
    nodes_label=comp_nodes+pred_nodes+var_nodes
    comp_color=comp_color_pred+comp_color_comp
    source_predL=np.zeros(np.shape(pred_yticks))
    source_compL=np.zeros(np.shape(var_yticks))
    for ii in range(len(pred_yticks)):
        source_predL[ii]=pred_y_nodes.index(pred_yticks[ii])+n_comp_nodes
    for ii in range(len(var_yticks)):
        source_compL[ii]=var_y_nodes.index(var_yticks[ii])+n_comp_nodes+len(pred_nodes)
    link_source=np.hstack((source_predL,vDom_comp))
    link_value=np.hstack((sPredVal,vDomVal))
    link_color=comp_color#sPred_color+vDom_color
    link_target=np.hstack((sPred_comp,source_compL))
    fig = go.Figure(data=[go.Sankey(
        arrangement = "perpendicular",
        node = dict(
          pad = 20,
          thickness = 4,
          line = dict(color = "gray", width = 0.2),
          label = nodes_label,
          color = "gray"
          ),
        link = dict(
          source = link_source, # indices correspond to labels, eg A1, A2, A1, B1, ...
          target = link_target,
          value = link_value,
          color = link_color
      ),
        hoverlabel=dict(
              align=nodes_label_align
      )
    )])

    fig.update_layout(title_text="Connection Diagram", font_size=10)
    fig.show()