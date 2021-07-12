# -*- coding: utf-8 -*-
"""
Created on Sun May 23 18:41:15 2021

@author: Bernardo
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from ggfp_v2 import ggfp
import sys
from scipy.spatial.distance import cdist
from fcm import fcm


#Import Iris dataset

df_full = pd.read_csv("iris.csv")
columns = list(df_full.columns)
features = columns[0:len(columns) - 1]
data = df_full[features]

# df_full = pd.read_csv("file.csv")
# columns = list(df_full.columns)
# features = columns[0:len(columns) -1]
# data = df_full[features]


# Simple dataset:
# data = pd.DataFrame([
#         [1,2], 
#         [2,3], 
#         [9,4], 
#         [10,1],])

data=data.to_numpy()



# Number of samples
n = len(data)  # the number of row
d = len(data[0]) #number of features/attributes
#d = len(data.columns)

    
#Define Focal point

#P=5*np.array([1, 1, 1]).reshape((1, 3))      #for the simple dataset
#P=np.array([5, 5, 5, 5, 10]).reshape((1, 5))  #for the iris dataset

#focal point using mean of data
datamean=np.array(np.mean(data,axis=0))
Pvalue=np.array([10])
P=np.append(datamean,Pvalue,axis=0)
P=np.reshape(P,(1,d+1))




#Extending X and Centers by adding null coordinates
w=np.shape(P)[1]     #nº of columns : dimension of P

if w>d:
    addZerosX = np.zeros((n, w-d))
    data = np.append(data, addZerosX, axis=1)
    
    
else:
    print("Focal point should be a higher dimension than data") 
    sys.exit()
     
    
    
 

   
 
def xie_beni_inv(U, C, nClusters,dist,distC):
    #sum_cluster_distance = 0
    minimum =math.inf
    #dist = cdist(C, data, metric='sqeuclidean')    
    num= np.sum(np.sum(np.multiply(U**m,dist)))    
    #distC=cdist(C, C, metric='sqeuclidean')    
    aux= np.min(distC[np.nonzero(distC)])
    
    if minimum>aux:
        minimum=aux
        
    XB_inv= n*minimum  / num 
    
    return XB_inv



def updateU(C, nClusters,dist,m): 
    #Update partition matrix U

    # dist = cdist(C, data, metric='sqeuclidean')   
        # #Calculate distance between data points and cluster centers


    aux = (1 / dist) ** (1 / (m-1))
    u = (aux / aux.sum(axis=0))
    
    return u
    return u
def typicallity(U):
    #Find the clusters that have typical data. A cluster is considered to have no typical data 
    # if for all points the U values are not the maximum in any point.
    
    indices=np.zeros((n,1), dtype=int)
    
    for i in range(n):
        
          ind = np.argmax(U[i])
          indices[i]=ind    
         
    indices=np.unique(indices)  # list of all clusters with typical datum

    return indices

"Run iterative algorithm"

zetashift=0.005 #zeta increment
zeta=1 #zeta
m=2 #fuzzy parameter
nClusters=10 # Initial number of clusters

#Some initialiations for needed variables
zetaVar=np.arange(0, zeta, zetashift)   #range of the iteration 0:zetashift:zeta
XB_inv= np.zeros((len(zetaVar), 1))    # initialize XB
ClusterNum=np.zeros((len(zetaVar), 1)) #auxiliar function to keep track of cluster number
numZeta=0


#C = np.zeros((nClusters,w))    #Initialize centers of clusters
#C = np.random.rand(nClusters,w)    #Initialize centers of clusters


for zetaIter in zetaVar:
    #Run the fcmfp algorithm
    C, U, labels, obj_fcn,dist,dist_FP,M,Pi,dist_C = ggfp(nClusters,zetaIter,m,init='fcm')   #GGFP(C,nClusters,zeta,m)
    
    #Remove negligle centers using the definition of typicallity
    indices = typicallity(U.T)
    C=C[indices] 
    
    #Get new number of clustesr
    nClusters=len(C)
    
    #Calculate new membership values
    U = updateU(C, nClusters,dist.T,m)
    
    #Calculate internal validity
    XB_inv[numZeta]=xie_beni_inv(U, C, nClusters,dist.T,dist_C.T)
    
    #keeping track of cluster number
    ClusterNum[numZeta] = nClusters 
    numZeta += 1
    print("Zeta:",zetaIter)    
    
"Plots"
fig, axs = plt.subplots(2,sharex=True)

axs[0].plot(zetaVar,XB_inv)
axs[0].set_title("Xie Beni inverted")
axs[1].plot(zetaVar,ClusterNum, 'r')
axs[1].set_title("Nº of Clusters")
fig.tight_layout()
# plt.plot(zetaVar,XB_inv)
# plt.plot(zetaVar,ClusterNum)        