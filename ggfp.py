# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:47:25 2021

@author: Bernardo
"""

import numpy as np, numpy.random
import pandas as pd
from scipy.spatial.distance import cdist
from sammon import sammon
import sys
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from rand_index import rand_index_score
from sklearn import datasets, preprocessing
from fcm import fcm
from kmeanspp import KMeans


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



def updateU(C, nClusters,dist,m): 
    #Update partition matrix U

    # dist = cdist(C, data, metric='sqeuclidean')   
        # #Calculate distance between data points and cluster centers


    aux = (1 / dist) ** (1 / (m-1))
    u = (aux / aux.sum(axis=0))
    
    return u

def labelData(u):
    #Find clustering results

    clustering = u.argmax(axis=0)
   
    return clustering

def obj_function(u,C,dist,dist_FP,zeta,m):

    # dist = cdist(C, data, metric='sqeuclidean') 
    # distFP=cdist(C, P, metric='sqeuclidean')

    aux=np.sum(dist_FP) 
    aux=np.nan_to_num(aux)
    
    obj_fcn =( np.sum(u**m * dist) + (zeta * aux) )*100  #obj function  (multiplicado por 100 porque os valores de A sao muito pequenos devido a extender a dimensao)
    
    obj_fcn=np.nan_to_num(obj_fcn)
                                        
    return obj_fcn

def project_centers(C,P):
    (C,index) = np.unique(C,axis=0,return_index=True)
    C=C+1e-10
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



 
       
def ggfp(data,P,nClusters,zeta,m, init, max_iter=100,min_impro=1e-6):
    w=np.shape(P)[1]
        # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes

    if init=='kmeans++':
     
        kmeans = KMeans(nClusters,init='k-means++', random_state=0).fit(data)
    
        
        C=kmeans.cluster_centers_     
      
        if w>d:
            #Extending X and Centers by adding null coordinates
            addZerosX = np.zeros((n, w-d))
            data = np.append(data, addZerosX, axis=1) 
            addZerosC = np.zeros((nClusters, w-d))
            C = np.append(C, addZerosC, axis=1)       #extend dimension of centers

        else: 
            print("Focal point should be a higher dimension than data") 
            sys.exit()
      
        
        #random inital partition matrix
        U = np.random.dirichlet(np.ones(nClusters),size=n).T
        #euclidean metric partition matrix (fcm)
        # dist = cdist(C, data, metric='sqeuclidean')   
        # aux = (1 / dist) ** (1 / (m-1))
        # U = (aux / aux.sum(axis=0))
        #print('kmeans here')

    elif init=='fcm':
        

        #generate seed for random numbers
        rng = np.random.default_rng(seed=42) #use np.random.default_rng() for random sequences


        C = rng.random((nClusters,len(data[0])))    #Initialize centers of clusters


        C, U, labels, obj_fcn = fcm(data,C,nClusters,m)   #fcm(C,nClusters,m)

        
        if w>d:
            #Extending X and Centers by adding null coordinates
            addZerosX = np.zeros((n, w-d))
            data = np.append(data, addZerosX, axis=1) 
            addZerosC = np.zeros((nClusters, w-d))
            C = np.append(C, addZerosC, axis=1)       #extend dimension of centers

        else: 
            print("Focal point should be a higher dimension than data") 
            sys.exit()
      
        # addZerosX = np.zeros((n, w-d))
        # data = np.append(data, addZerosX, axis=1)
        # #print('fcm here')
       
    elif init=='random':

        #generate seed for random numbers
        rng = np.random.default_rng(seed=42) #use np.random.default_rng() for random sequences


        C = rng.random((nClusters,d))    #Initialize centers of clusters

        if w>d:
            #Extending X and Centers by adding null coordinates
            addZerosX = np.zeros((n, w-d))
            data = np.append(data, addZerosX, axis=1) 
            addZerosC = np.zeros((nClusters, w-d))
            C = np.append(C, addZerosC, axis=1)       #extend dimension of centers

        else: 
            print("Focal point should be a higher dimension than data") 
            sys.exit()        
        #C = np.random.rand(nClusters,d)    #Initialize centers of clusters
        
        U = rng.dirichlet(np.ones(nClusters),size=n).T
   
    
   #GGFP algorithm
    obj_fcn = np.zeros((max_iter, 1))
    Pi=np.zeros((nClusters,1))
    dist=np.zeros((n,nClusters))
    dist_C=np.zeros((nClusters,nClusters))
    it = 0
    for it in range(max_iter):
        
        M=[]
        sumfm=np.sum(np.power(U,m), axis=1, keepdims=True)

        for i in range(nClusters):
            
            A=np.zeros((w,w))
            xv=data-C[i,:]
            Pv=P-C
            
            #Covariance matrix calculation
            a=np.dot(np.power(U[i,:],m) * np.transpose(xv), xv)
            b=zeta*np.dot(Pv.T,Pv)

            A=np.divide(( a + b ), (sumfm[i,:] + zeta))  
            A=A+np.identity(w)*0.0001 #Covariance matrix regularization
            M.append(A)
            
            #Priori probability
            Pi[i,:]=1/n*sumfm[i,:]
            
            
            #Calculate distance between data points and cluster centers
            dist_aux=(1/(np.linalg.det(np.linalg.pinv(A))**0.5) * 1/Pi[i,:] 
              * np.exp(0.5*np.sum((np.dot(xv,np.linalg.pinv(A))*xv), axis=1,keepdims=True)))            
            dist_aux=dist_aux+1e-10
            dist[:,i]=dist_aux[:,0]
            dist=np.nan_to_num(dist)
            
            #Calculate distance between Focal Point and cluster centers       
            dist_FP=(1/(np.linalg.det(np.linalg.pinv(A))**0.5) * 1/Pi[i,:] 
                * np.exp(0.5*np.sum((np.dot(Pv,np.linalg.pinv(A))*Pv), axis=1,keepdims=True)))
            
            dist_FP=dist_FP + 1e-10
            dist_FP=np.nan_to_num(dist_FP)
            
            #Calculate distance between the cluster centers (for iterative algorithm)       
            # dist_auxC=(1/(np.linalg.det(np.linalg.pinv(A))**0.5) * 1/Pi[i,:] 
            #     * np.exp(0.5*np.sum((np.dot(C,np.linalg.pinv(A))*C), axis=1,keepdims=True)))
            
            # dist_auxC=dist_auxC + 0.001
            # dist_C[:,i]=dist_auxC[:,0]

            # dist_C=np.nan_to_num(dist_C)
   


        #dist=calculateDistance(nClusters,M,Pi)
        #dist_FP=calculateDistanceFP(nClusters,M,Pi)
        U = updateU(C, nClusters,dist.T,m)
        C = calculateCenters(data,U,P,nClusters,zeta,m)

        obj_fcn[it]=obj_function(U,C,dist.T,dist_FP.T,zeta,m)
             
            
        if it > 1:
            
            if abs(obj_fcn[it] - obj_fcn[it-1]) < min_impro:  #If objective function does not show improvement stop running algorithm
                
                break
            
                           
        it += 1

    label = labelData(U)    
    projCenter= project_centers(C,P)
    
    return  C, U, label,obj_fcn,dist,dist_FP,M,Pi,dist_C,projCenter





"#Call and run the ggfp algorithm"

if __name__ == '__main__':
    
    #Choose number of clusters and dataset
    nClusters=3    
    dataset='iris'   #'iris','bearing'
    
    #load dataset
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


    #  Call GGFP : ... =GGFP(data,P,nClusters,zeta,m,init='fcm','kmeans++')
    C, U, labels, obj_fcn,dist,dist_FP,M,Pi,dist_C,projCenter = ggfp(data,P,nClusters,0.1,3,init='fcm')   

    #"Adjusted Rand Index"
    #adjusted_rand_scoreGGFP=adjusted_rand_score(label_gt, labels)
    
    # print('GGFP Adjusted Rand Index :', adjusted_rand_score(label_gt, labels))





