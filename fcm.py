# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:28:51 2021

@author: Bernardo
"""

import numpy as np, numpy.random
import pandas as pd
from scipy.spatial.distance import cdist
from sammon import sammon
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import datasets
from rand_index import rand_index_score




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


def calculateCenters(data,u,nClusters,m):
    #Calculates centers of clusters

    num=np.dot((u**m), data) 
    
    den=np.sum(np.power(u,m), axis=1, keepdims=True)
    
    C =  num/den

    return C




def updateU(data,C, nClusters,m): 
    #Calculate partition matrix U
    
    dist = cdist(C, data, metric='sqeuclidean')   
    aux = (1 / dist) ** (1 / (m-1))
    u = (aux / aux.sum(axis=0))
    
    return u


def labelData(u):
    #Find clustering results
    clustering = u.argmax(axis=0)
   
    return clustering

def obj_function(data,u,C,m):

    dist = cdist(C, data, metric='sqeuclidean') 
    
             
    obj_fcn = np.sum(u**m * dist)   #obj function  
        
                                            
    return obj_fcn




    
def fcm(data,C,nClusters,m, max_iter=100,min_impro=1e-6):
    #FCM algorithm
    obj_fcn = np.zeros((max_iter, 1))
    it = 0

    for it in range(max_iter):
        U = updateU(data,C, nClusters,m)
        C = calculateCenters(data,U,nClusters,m)
  
        obj_fcn[it]=obj_function(data,U,C,m)
             
            
        if it > 1:
            
            if abs(obj_fcn[it] - obj_fcn[it-1]) < min_impro:  #If objective function does not have any more improvement stop running algorithm
                
                break
            
                           
        it += 1

    label = labelData(U)
    return  C, U, label,obj_fcn




"#Call and run the fcm algorithm"

if __name__ == '__main__':
    #Choose number of clusters and dataset
    nClusters=3    
    dataset='iris'   #'iris','bearing'
    
    #load data
    data, label_gt =load_data(dataset)
    # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes

    #generate seed for random numbers
    rng = np.random.default_rng(seed=42) #use np.random.default_rng() for random  
    #C=np.zeros(nClusters,d)          #Initialize centers of clusters
    C = rng.random((nClusters,d))   
    
    #call fcm
    C, U, labels, obj_fcn = fcm(data,C,nClusters,2)   #FCM(C,nClusters,m)


    #adjusted_rand_scoreFCM=adjusted_rand_score(label_gt, labels)
    
    #print('\nFCM Adjusted Rand Index :',adjusted_rand_score(label_gt, labels) )


