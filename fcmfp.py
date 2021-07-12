# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:28:10 2021

@author: Bernardo
"""


import numpy as np, numpy.random
import pandas as pd
from scipy.spatial.distance import cdist
from sammon import sammon
import sys
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import datasets, preprocessing



def load_data(dataset):
    
    if dataset=='iris':
        "Iris data"
        iris = datasets.load_iris()
        data = iris.data 
        label_iris = iris.target
        label_gt=label_iris
        
    elif dataset=='bearing':
                
        "Bearing data"
        data = pd.read_csv("bearingdata.csv",header=None)        
        data=data.to_numpy()
        
        label_bearing = pd.read_csv("bearingdata_label.csv", header=None)
        label_bearing=label_bearing.to_numpy().T
        label_bearing=label_bearing.reshape(len(label_bearing))
        
        label_gt=label_bearing #ground truth label

    return data , label_gt






def calculateCenters(data,u,P,nClusters,zeta,m):
    #Calculates centers of clusters

    zetaP = np.dot(zeta,P)

    num=np.dot((u**m), data) + zetaP
    
    den=np.sum(np.power(u,m), axis=1, keepdims=True)   + zeta 
    
    C =  num/den
    #C=C[np.mean(C,axis=1).argsort()]  #organize the centers by mean of the rows, from closest to origin to furthest

    return C



def updateU(data,C, nClusters,m): 
    #Update partition matrix U

    dist = cdist(C, data, metric='sqeuclidean')   
    aux = (1 / dist) ** (1 / (m-1))
    u = (aux / aux.sum(axis=0))
    
    return u

def labelData(u):
    #Find clustering results

    clustering = u.argmax(axis=0)
   
    return clustering

def obj_function(data, u,C,P,zeta,m):

    dist = cdist(C, data, metric='sqeuclidean') 
    distFP=cdist(C, P, metric='sqeuclidean')
             
    obj_fcn = np.sum(u**m * dist) + (zeta * np.sum(distFP))   #obj function  
        
                                            
    return obj_fcn

def project_centers(C,P):
    C_col=len(C[0])  #nr of columns of center matrix = nº of features in data
    C_row=len(C)        #nr of rows = nr of clusters
    P_dim=len(P[0])    #nr of features 
    projectedCenter=np.zeros((C_row,C_col-1))
    
    for i in range(C_row):
        vDirector = C[i, :] - P
        #C[i, C_col - 1] = 0
        t =  - P[:,P_dim-1] / vDirector[:,len(vDirector[0])-1]
        for j in range(C_col-1):
            projectedCenter[i, j] = P[0,j] + np.multiply(t , vDirector[0,j])
            

    return projectedCenter


    
def fcmfp(data,P, C,nClusters,zeta,m, max_iter=100,min_impro=1e-6):
    w=np.shape(P)[1]
        # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes
    if w>d:
        addZerosX = np.zeros((n, w-d))
        data = np.append(data, addZerosX, axis=1)   
    
    else: 
        print("Focal point should be a higher dimension than data") 
        sys.exit()
      
    #FCMFP algorithm
    obj_fcn = np.zeros((max_iter, 1))
    it = 0

    for it in range(max_iter):
        U = updateU(data,C, nClusters,m)
        C = calculateCenters(data, U,P,nClusters,zeta,m)

        obj_fcn[it]=obj_function(data, U,C,P,zeta,m)
             
            
        if it > 1:
            
            if abs(obj_fcn[it] - obj_fcn[it-1]) < min_impro:  #If objective function does not show improvement stop running algorithm
                
                break
            
                           
        it += 1

    label = labelData(U)    
    projCenter= project_centers(C,P)
    
    return  C, U, label,obj_fcn, projCenter

"#Call and run the fcmfp algorithm"
if __name__ == '__main__':
    #Choose number of clusters and dataset
    nClusters=3    
    dataset='iris'   #'iris','bearing'
    
    #load data
    data, label_gt =load_data(dataset)
    # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes

    
    "Define Focal point"
    
    #P=5*np.array([1, 1, 1]).reshape((1, 3))      #for the simple dataset
    #P=np.array([1, 1, 1, 1, 1]).reshape((1, 5))  #for the iris dataset
    
    #focal point using mean of data
    datamean=np.array(np.mean(data,axis=0))
    Pvalue=np.array([10])
    P=np.append(datamean,Pvalue,axis=0)
    P=np.reshape(P,(1,d+1))
    
    w=np.shape(P)[1]     #nº of columns : dimension of P

    #generate seed for random numbers
    rng = np.random.default_rng(seed=42) #use np.random.default_rng() for random    
    C = rng.random((nClusters,w))    #Initialize randomly centers of clusters    
    # C = np.zeros(nClusters,w)    #use zeros as intial centers
    
    C, U, labels, obj_fcn, projCenter = fcmfp(data,P, C,nClusters,1,2)   #FCMFP(C,nClusters,zeta,m)

    #"Adjusted Rand Index"
    #adjusted_rand_scoreFCMFP=adjusted_rand_score(label_gt, labels)
    
    # print('FCMFP Adjusted Rand Index :', adjusted_rand_score(label_gt, labels))


