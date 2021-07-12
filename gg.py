# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:04:18 2021

@author: Bernardo
"""

import numpy as np, numpy.random
import pandas as pd

from sammon import sammon
import matplotlib.pyplot as plt
from fcm import fcm
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import datasets
from kmeanspp import KMeans
from sklearn import preprocessing


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




#Calculates centers of clusters
def calculateCenters(data,u,nClusters,m):
    

    num=np.dot((u**m), data) 
    
    den=np.sum(np.power(u,m), axis=1, keepdims=True)
    
    C =  num/den
    

   #C=C[np.mean(C,axis=1).argsort()]  #organize the centers by mean of the rows, from closest to origin to furthest
      
    return C

#Update partition matrix U
def updateU(C, nClusters,dist,m): 
    
    #dist = cdist(C, data, metric='sqeuclidean')   
    aux = (1 / dist) ** (1 / (m-1))
    u = (aux / aux.sum(axis=0))
    
    return u

def obj_function(u,C,dist,m):

    obj_fcn = np.sum(u**m * dist)   #obj function  
        
                                            
    return obj_fcn

def labelData(u):
    
    clustering = u.argmax(axis=0)
   
    return clustering




def gg(data, nClusters,m,init,max_iter=100,min_impro=1e-6):

        # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes

    
    if init=='kmeans++':
         
        kmeans = KMeans(nClusters,init='k-means++', random_state=0).fit(data)
        
            
        C=kmeans.cluster_centers_
        U = np.random.dirichlet(np.ones(nClusters),size=n).T
        # dist = cdist(C, data, metric='sqeuclidean')   
        # aux = (1 / dist) ** (1 / (m-1))
        # U = (aux / aux.sum(axis=0))
        #print('kmeans here')

    elif init=='fcm':
        #generate seed for random numbers
        rng = np.random.default_rng(seed=42) #use np.random.default_rng() for random sequences


        C = rng.random((nClusters,d))    #Initialize centers of clusters

        
        #C = np.random.rand(nClusters,d)    #Initialize centers of clusters
        C, U, labels, obj_fcn = fcm(data, C,nClusters,m)   #fcm(C,nClusters,m)
        #print('fcm here')
   
    elif init=='random':
    
        #generate seed for random numbers
        rng = np.random.default_rng(seed=42) #use np.random.default_rng() for random sequences


        C = rng.random((nClusters,d))    #Initialize centers of clusters

        
        #C = np.random.rand(nClusters,d)    #Initialize centers of clusters
        
        U = rng.dirichlet(np.ones(nClusters),size=n).T


    
    obj_fcn = np.zeros((max_iter, 1))
    it = 0


    dist=np.zeros((n,nClusters))
    Pi=np.zeros((nClusters,1))
    # M= [[]for x in range(nClusters)]

    for it in range(max_iter):
        
        sumfm=np.sum(np.power(U,m), axis=1, keepdims=True)
        
        M=[]
        for i in range(nClusters):
            xv=data-C[i,:]
            #Covariance matrix calculation
            A=np.divide(np.dot(np.power(U[i,:],m) * np.transpose(xv), xv) , sumfm[i,:]) 
            Pi[i,:]=1/n*sumfm[i,:]
            A=A+np.identity(d)*0.0001


            M.append(A)
            dist_aux=(1/(np.linalg.det(np.linalg.pinv(A))**0.5) * 1/Pi[i,:] 
            * np.exp(0.5*np.sum((np.dot(xv,np.linalg.pinv(A))*xv), axis=1,keepdims=True)))
            dist_aux=np.nan_to_num(dist_aux)

            dist_aux=dist_aux+1e-10
            dist[:,i]=dist_aux[:,0]

        U = updateU(C, nClusters,dist.T,m)  
        C = calculateCenters(data,U,nClusters,m)
        
        obj_fcn[it]=obj_function(U,C,dist.T,m)
            
        if it > 1:
            
            if abs(obj_fcn[it] - obj_fcn[it-1]) < min_impro:  #If objective function does not have any more improvement stop running algorithm
                
                break
            
                           
        it += 1

    
    labels= labelData(U) 
   
    return U, C, A,labels, dist, obj_fcn,Pi,M

if __name__ == '__main__':
    #Choose number of clusters and dataset
    nClusters=7    
    dataset='iris'   #'iris','bearing'
    
    #load data
    data, label_gt =load_data(dataset)
    # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes
    U, C, A,labels, dist, obj_fcn,Pi ,M=gg(data,nClusters,3,init='fcm')  #gg(data,nClusters,m,init='fcm','kmeans++')
    


    #print('GG Adjusted Rand Index :', adjusted_rand_score(label_gt, labels))


    #adjusted_rand_scoreGG=adjusted_rand_score(label_gt, labels)


    #print('GG Adjusted Rand Index :', adjusted_rand_score(label_gt, labels))

