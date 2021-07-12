# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:33:32 2021

@author: Bernardo
"""
import numpy as np
import pandas as pd
from fcm import fcm
from fcmfp import fcmfp
from gg import gg
from ggfp import ggfp
from sklearn import datasets
from sklearn.metrics.cluster import adjusted_rand_score
from rand_index import rand_index_score
import sys
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

nRuns=30

nClusters=7

dataset='bearing'     


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



#generate seed for random numbers
rng = np.random.default_rng() #use np.random.default_rng() for random sequence

def labelData(u):
    #Find clustering results
    clustering = u.argmax(axis=0)
   
    return clustering
#Load the data and the gt label
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


"#Run FCM"
RI_fcm=np.zeros((nRuns))
ARI_fcm=np.zeros((nRuns))

for i in range(nRuns):
    data, label_gt =load_data(dataset)
    # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes
    # #data = iris.data 
    # data, label_gt =load_data(dataset)
    # n = len(data)  # the number of row
    # d = len(data[0]) #number of features/attributes

    # C = rng.random((nClusters,d))    #Initialize centers of clusters
    # C, U, labels, obj_fcn = fcm(C,nClusters,2)   #FCM(C,nClusters,m)

    
    #C=np.zeros(nClusters,d)          #Initialize centers of clusters
    C = rng.random((nClusters,d))   
    
    C, U, labels, obj_fcn = fcm(data,C,nClusters,2)   #FCM(C,nClusters,m)

    adjusted_rand_score_FCM=adjusted_rand_score(label_gt, labels)
    rand_index_FCM=rand_index_score(label_gt, labels)
    
    RI_fcm[i]=rand_index_FCM
    ARI_fcm[i]=adjusted_rand_score_FCM
    
    
    
"#Run FCMFP"
RI_fcmfp=np.zeros((nRuns))
ARI_fcmfp=np.zeros((nRuns))

for i in range(nRuns):
    data, label_gt =load_data(dataset)
    # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes
    #generate seed for random numbers
    rng = np.random.default_rng(seed=42) #use np.random.default_rng() for random
    
    C = rng.random((nClusters,w))    #Initialize centers of clusters
    
    C, U, labels, obj_fcn, projCenter = fcmfp(data,P, C,nClusters,1,2)   #FCMFP(C,nClusters,zeta,m)

    

    adjusted_rand_score_fcmfp=adjusted_rand_score(label_gt, labels)
    rand_index_fcmfp=rand_index_score(label_gt, labels)
    
    RI_fcmfp[i]=rand_index_fcmfp
    ARI_fcmfp[i]=adjusted_rand_score_fcmfp
 

"#Run GG"
RI_gg=np.zeros((nRuns))
ARI_gg=np.zeros((nRuns))

for i in range(nRuns):
    data, label_gt =load_data(dataset)
    # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes
    # #data = iris.data 
    # data, label_gt =load_data(dataset)

    # n = len(data)  # the number of row
    # d = len(data[0]) #number of features/attributes

    U, C, A,labels, dist, obj_fcn,Pi ,M=gg(data,nClusters,2,init='fcm')  #gg(data,nClusters,m,init='fcm','kmeans++')


    
    adjusted_rand_score_gg=adjusted_rand_score(label_gt, labels)
    rand_index_gg=rand_index_score(label_gt, labels)
    
    RI_gg[i]=rand_index_gg
    ARI_gg[i]=adjusted_rand_score_gg
    
" #Run GGFP"
RI_ggfp=np.zeros((nRuns))
ARI_ggfp=np.zeros((nRuns))

for i in range(nRuns):
    data, label_gt =load_data(dataset)
    # Number of samples
    n = len(data)  # the number of row
    d = len(data[0]) #number of features/attributes
    
    #  GGFP(data,P,nClusters,zeta,m,init='fcm','kmeans++')
    C, U, labels, obj_fcn,dist,dist_FP,M,Pi,dist_C,projCenter = ggfp(data,P,nClusters,0.1,3,init='fcm')   

    adjusted_rand_score_ggfp=    adjusted_rand_score(label_gt, labels)

    rand_index_ggfp=rand_index_score(label_gt, labels)
    
    RI_ggfp[i]=rand_index_ggfp
    ARI_ggfp[i]=adjusted_rand_score_ggfp
           
    
    
    
    
 
"Plot ARI"
data_plot_ARI = [ARI_fcm, ARI_fcmfp, ARI_gg, ARI_ggfp] 
fig, ax = plt.subplots(figsize=(12, 7))
# Remove top 
ax.spines['top'].set_visible(False)
# Add major gridlines in the y-axis
ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
# Set  names as labels for the boxplot
xlabel = ['FCM',' FCMFP', 'GG',' GGFP'] 
# Creating plot
bp = ax.boxplot(data_plot_ARI,labels=xlabel)
# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color ='red',
               linewidth = 2.5)
# Adding title
plt.title("ARI Boxplot") 
# show plot
plt.show()


"Plot RI"
data_plot_RI = [RI_fcm, RI_fcmfp, RI_gg, RI_ggfp] 
fig2, ax2 = plt.subplots(figsize=(12, 7))
# Remove top 
ax2.spines['top'].set_visible(False)
# Add major gridlines in the y-axis
ax2.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
# Set  names as labels for the boxplot
xlabel2 = ['FCM',' FCMFP', 'GG',' GGFP'] 
# Creating plot
bp2 = ax2.boxplot(data_plot_RI,labels=xlabel2)
# changing color and linewidth of
# medians
for median in bp2['medians']:
    median.set(color ='red',
               linewidth = 2.5)
# Adding title
plt.title("RI Boxplot") 
# show plot
plt.show()


"Wilcoxon Test"
#significance level  α = 0.05 (confidence interval of 95%), i.e., if
# p-value < 0.05 it is considered that there is a statistical significant difference among the results being
# there is no statistical significant difference among results,
#otherwise

            
w, p1 = wilcoxon(ARI_fcm, ARI_fcmfp, zero_method='zsplit')
w, p2 = wilcoxon(ARI_fcm, ARI_gg, zero_method='zsplit')
w, p3 = wilcoxon(ARI_fcm, ARI_ggfp, zero_method='zsplit')
w, p4 = wilcoxon(ARI_fcmfp, ARI_gg, zero_method='zsplit')
w, p5 = wilcoxon(ARI_fcmfp, ARI_ggfp, zero_method='zsplit')
w, p6 = wilcoxon(ARI_gg, ARI_ggfp, zero_method='zsplit')



#Create a dataframe with all the p values
wilcoxon_test = pd.DataFrame.from_dict({
    'FCM,FCMFP': [p1],
    'FCM,GG': [p2],
    'FCM,GGFP': [p3],
    'FCMFP,GG': [p4],
    'FCMFP,GGFP': [p5],
    'GG,GGFP': [p6]
},
orient='index', columns=['p'])


    
    