import numpy as np
import pandas as pd

def read_file(f, Z_col, G_col):
    file = pd.read_csv(f)
    name=list(file)
    ZGlist=list()
    Z_label=list()
    G_label=list()
    Label={}
    file = file.replace([-88, -77, -99], np.nan)
    for i in range(len(Z_col)):
        list.append(ZGlist,file[file.columns[Z_col[i]]])
        for j in range(len(Z_col[i])):
            list.append(Z_label,name[(Z_col[i])[j]])
    for i in range(len(G_col)):
        list.append(ZGlist,file[file.columns[G_col[i]]])
        for j in range(len(G_col[i])):
            list.append(G_label,name[(G_col[i])[j]])
    dataZG = pd.concat(ZGlist, axis = 1)
    ZG = dataZG.dropna()
    Z = ZG[ZG.columns[0:len(Z_label)]]
    G = ZG[ZG.columns[len(Z_label):len(Z_label)+len(G_label)]]
    Z = Z.to_numpy()
    G = G.to_numpy()
    Z = Z.astype(np.float64)
    G = G.astype(np.float64)
    Label['Z_label']=Z_label
    Label['G_label']=G_label
    return Z, G, Label