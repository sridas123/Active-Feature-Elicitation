# -*- coding: utf-8 -*-
"""

Created on Fri Jun 16 21:00:50 2017

@author: Srijita
"""
from __future__ import division # floating point division
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import random


"""Windows Machine path"""
#path="C:\\Users\\Srijita\\Dropbox\\Active_feature_acquisition\\Data\\Cardia\\y20\\"

obs_data_indx="obs_data_indx.txt"
unobs_data_ranking="unobs_idx_rank_data.txt"
test_data_indx="test_data_indx.txt"

"""Linux Machine path"""
#path="C:\\Users\\sridas\\Dropbox\\Active_Feature_Acquisition\\Data\\"

"""Office Machine path"""
#path="C:\\Users\\sxd170431\\Dropbox\\Active_feature_acquisition\\Data\\Cardia\\y20\\"

"""This specifies the observed and the unobserved dataset size"""
#obs_size=50
#unobs_size=1000
#test_size=100

"""This specifies the number of features that are observed"""
#obs_feat_num=8

#data_name="train_facts_year20_clean"

#obsrvd_data_full=path+data_name+"_obs.csv"
#unobsrvd_data_full=path+data_name+"_unobs_all.csv"
#unobsrvd_data=path +data_name+ "_unobs.csv"
#test_data=path+data_name+"_test.csv"

#observed_feat_names=[1,2,3,4,5,6,7]
#pos_to_neg=0

def splitdataset(dataset,obs_size,unobs_size,test_size):
        
    print "I am trying to split the dataset now"
    randindices = np.random.randint(0,dataset.shape[0],obs_size+unobs_size+test_size)
    obs = dataset[randindices[0:obs_size],0:dataset.shape[1]]
    unobs = dataset[randindices[obs_size:obs_size+unobs_size],0:dataset.shape[1]]
    test = dataset[randindices[obs_size+unobs_size:obs_size+unobs_size+test_size],0:dataset.shape[1]]
    
    return (obs,unobs,test)

"""Sample from the Observed data pool and testdata pool"""
def splitdataset_pool(obspool_ind,testpool_ind,dataset,obs_size,test_size):
        
    print "I am trying to split the dataset"
    all_ind=range(0,dataset.shape[0])
    #randindices = np.random.randint(0,dataset.shape[0],obs_size+test_size)
    print obspool_ind[0:10]
    obs_ind=random.sample(obspool_ind,obs_size)
    test_ind=random.sample(testpool_ind,test_size)
    #print obs_ind
    obs = dataset[obs_ind,0:dataset.shape[1]]
    test = dataset[test_ind,0:dataset.shape[1]]
    
    unobs_ind=[]
    for i in all_ind:
        if (i not in obspool_ind) and (i not in testpool_ind):
           unobs_ind.append(i)
   
    unobs = dataset[unobs_ind,0:dataset.shape[1]]
           
    return (obs,unobs,test)

"""This function is for the Active Learning setting"""
def splitdataset_act(dataset, obs_size):
        
    print "I am trying to split the dataset now for Active Learning"
    """Do an equal sampling of positive and negative class in the Observed set"""
    label_indx=dataset.shape[1]-1
    datapos=dataset[dataset[:,label_indx]==1]
    dataneg=dataset[dataset[:,label_indx]==0]
    randindices_pos = np.random.randint(0,datapos.shape[0],int(obs_size/2))
    obs_pos=datapos[randindices_pos,0:datapos.shape[1]]
    randindices_neg = np.random.randint(0,dataneg.shape[0],int(obs_size/2))
    obs_neg=dataneg[randindices_neg,0:dataneg.shape[1]]
    obs=np.vstack((obs_pos,obs_neg))
    obs_id=obs[:,0]
    #obs = dataset[randindices,0:dataset.shape[1]]
    unobs=[]
    for i in range(0,dataset.shape[0]):
        if dataset[i][0] not in obs_id:
           unobs.append(dataset[i])
           
    unobs=np.asarray(unobs)       
    return (obs,unobs)

def splitdataset_act1(dataset, obs_size):
        
    print "I am trying to split the dataset now for Active Learning"
    """Do an equal sampling of positive and negative class in the Observed set"""
    all_ind=range(0,dataset.shape[0])
    randindices = np.random.randint(0,dataset.shape[0],obs_size)
    obs=dataset[randindices,0:dataset.shape[1]]
    unobs_ind=[]
    for i in all_ind:
        if i not in randindices:
           unobs_ind.append(i) 
           
    unobs = dataset[unobs_ind,0:dataset.shape[1]]              
    return (obs,unobs)

def splitdataset_new(dataset,obs_size,test_size):
        
    print "I am trying to split the dataset"
    all_ind=range(0,dataset.shape[0])
    """Changing this for some experimentation; temporary change"""
    randindices = np.random.randint(0,dataset.shape[0],obs_size+test_size)
    obs = dataset[randindices[0:obs_size],0:dataset.shape[1]]
    test = dataset[randindices[obs_size:obs_size+test_size],0:dataset.shape[1]]
    
    unobs_ind=[]
    for i in all_ind:
        if i not in randindices:
           unobs_ind.append(i)
   
    unobs = dataset[unobs_ind,0:dataset.shape[1]]
           
    return (obs,unobs,test)

def splitdataset_fold(dataset,obs_size,test_size):
        
    print "I am trying to split the dataset"
    all_ind=range(0,dataset.shape[0])
    """Changing this for some experimentation; temporary change"""
    randindices = np.random.randint(0,dataset.shape[0],obs_size+test_size)
    obs = dataset[randindices[0:obs_size],0:dataset.shape[1]]
    test = dataset[randindices[obs_size:obs_size+test_size],0:dataset.shape[1]]
    
    unobs_ind=[]
    for i in all_ind:
        if i not in randindices:
           unobs_ind.append(i)
   
    unobs = dataset[unobs_ind,0:dataset.shape[1]]
           
    return (obs,unobs,test)

def return_train_label(data,start,end):
    
    train= data[:,start:end]
    label= data[:,end]
       
    return train,label
    
def splitdataset1(dataset, obs_size,unobs_size,test_size,path,obs_data_indx,unobs_data_ranking,test_data_indx,unobs_data_idx):
   
    obs_idx=np.loadtxt(path+obs_data_indx)
    test_idx=np.loadtxt(path+test_data_indx)
    unobs_idx=np.loadtxt(path+unobs_data_idx)
    obs_idx=obs_idx.tolist()
    test_idx=test_idx.tolist()
    unobs_idx=unobs_idx.tolist()
    print "The length of unobs_idx is", len(unobs_idx)
    obs_data=[]
    unobs_data=[]
    test_data=[]
    
    for i in range(0,dataset.shape[0]):
        if dataset[i][0] in obs_idx:
           obs_data.append(dataset[i])
        if dataset[i][0] in test_idx:
             test_data.append(dataset[i])
        if dataset[i][0] in unobs_idx:     
           unobs_data.append(dataset[i])
    print "The length of unobs_data is", len(unobs_data)   
    
    """The three datasets that are required"""
    obs_data=np.asarray(obs_data)
    unobs_data=np.asarray(unobs_data)     
    test_data=np.asarray(test_data)
    
    #randindices = np.random.randint(0,unobs_data.shape[0],unobs_size)
    #unobs = unobs_data[randindices,0:unobs_data.shape[1]]
    #randindices1 = np.random.randint(0,test_data.shape[0],test_size)
    """keep the test set fixed"""
    test = test_data[0:test_size,0:test_data.shape[1]]
    
    return (obs_data,unobs_data,test)

"""Function that splits data into train and test and returns them"""
def create_train_test_split(data,test_split):
    
    start=0;end=data.shape[1]-1
    data_train,data_label=return_train_label(data,start,end)
    testsize=(1/test_split)
    datapos_train, datapos_test, datapos_label_train, datapos_label_test = train_test_split(data_train, data_label, test_size=testsize)
    datapos_label_train=np.reshape(datapos_label_train,(-1,1))
    datapos_label_test=np.reshape(datapos_label_test,(-1,1))
    train=np.hstack((datapos_train,datapos_label_train))
    test=np.hstack((datapos_test,datapos_label_test))
    print "The train test split is", train.shape[0],test.shape[0]
    return train,test

def create_train_test_split1(data,test_size):
    
    start=0;end=data.shape[1]-1
    all_ind=range(0,data.shape[0])
    data_train,data_label=return_train_label(data,start,end)
    randindices=np.random.randint(0,data.shape[0],test_size)
    test = data[randindices,0:data.shape[1]]
    train_ind=[]
    for i in all_ind:
        if i not in randindices:
           train_ind.append(i)
    train = data[train_ind,0:data.shape[1]]
    return train,test

def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset
    
def write_csv(filename,data,start,end):
    with open(filename, 'wb') as csvfile:

     csvwriter = csv.writer(csvfile, delimiter=',')
     for i in range(0,data.shape[0]):
         if (end==data.shape[1]):
            csvwriter.writerow(data[i,start:end])
         else:
            col_ind=range(start,end)
            col_ind.append(data.shape[1]-1)
            csvwriter.writerow(data[i,col_ind])
"""This module is for the active feature acquisition code which does a model update after obtaining each point"""    
def process_dataset_act(obs_size, test_split,obsrvd_data_full,unobsrvd_data_full,test_data,unobsrvd_data,obs_unobs_all,path,data_name,obs_feat_num,pos_to_neg,first):  
    
    print "I am processing the active learning split dataset"
    filename=path+data_name+'.csv'
    dataset = loadcsv(filename)
    print "The shape is", dataset.shape
    """This is the sampling step in case someone wants to sample"""
    if ((pos_to_neg !=0) and (first==0)):
       dataset1=dataset 
       label_indx=dataset.shape[1]-1
       datapos=dataset[dataset[:,label_indx]==1]
       dataneg=dataset[dataset[:,label_indx]==0]
       
       """Pick a subset of positive class for creating the Train and Test set"""
       datapos_train,datapos_test=create_train_test_split(datapos,test_split)
       #datapos_train,datapos_test=create_train_test_split1(datapos,(int)(test_size/2))
       print "The shape of positive train and test data is",datapos_train.shape,datapos_test.shape
       
       posnum=datapos.shape[0]
       negnum=posnum* pos_to_neg
       randindices = np.random.randint(0,dataneg.shape[0],negnum)
       print "The number of points to be taken", len(randindices)
       negdata=dataneg[randindices,0:dataneg.shape[1]]
       
       """Pick a subset of negative class for creating the Train and Test set"""
       dataneg_train,dataneg_test=create_train_test_split(negdata,test_split)
       #dataneg_train,dataneg_test=create_train_test_split1(negdata,(int)(test_size/2))
       print "The shape of negative train and test data is",dataneg_train.shape,dataneg_test.shape
       
       dataset=np.vstack((datapos_train,dataneg_train))
       test=np.vstack((datapos_test,dataneg_test))
       
       """This method is used to store all data points except the test data"""
       test_indices=test[:,0]
       rest_but_test=dataset1[:,0]
       train_indices=list(set(rest_but_test) - set(test_indices))
       dataset2=[]
       for i in range(0,dataset1.shape[0]):
           if dataset1[i][0] in train_indices:
              dataset2.append(dataset1[i])
       dataset2=np.asarray(dataset2)       
       start=0
       end=dataset2.shape[1]
       write_csv(obs_unobs_all,dataset2,start,end)
       
    elif  ((pos_to_neg !=0) and (first!=0)):  
       train= loadcsv(obs_unobs_all) 
       label_indx=train.shape[1]-1
       trainpos=train[tain[:,label_indx]==1]
       trainneg=dataset[train[:,label_indx]==0]
    else:
       if first==0: 
          train,test=create_train_test_split(dataset,test_split) 
          start=0
          end=train.shape[1]
          write_csv(obs_unobs_all,train,start,end)
       else:
          train= loadcsv(obs_unobs_all) 
          print "I am loading"
       #train,test=create_train_test_split1(dataset,test_size)
       dataset=train
    
    print "The shape of dataset being considered", dataset.shape[0]      
    #obs,unobs = splitdataset_act1(dataset,obs_size)
    obs,unobs = splitdataset_act(dataset,obs_size)
    start=0
    end=obs.shape[1]
    write_csv(obsrvd_data_full,obs,start,end)
    write_csv(unobsrvd_data_full,unobs,start,end)
    if first==0:
       end=test.shape[1]
       write_csv(test_data,test,start,end)
    end=obs_feat_num
    write_csv(unobsrvd_data,unobs,start,end)
    
def create_pool(obspool_size,testpool_size,path,data_name,pos_to_neg):
    filename=path+data_name+'.csv'
    dataset = loadcsv(filename)
#    if (pos_to_neg !=0):
#       label_indx=dataset.shape[1]-1
#       datapos=dataset[dataset[:,label_indx]==1]
#       dataneg=dataset[dataset[:,label_indx]==0]
#       posnum=datapos.shape[0]
#       negnum=posnum* pos_to_neg
#       randindices = np.random.randint(0,dataneg.shape[0],negnum)
#       print "The length is", len(randindices)
#       negdata=dataneg[randindices,0:dataneg.shape[1]]
#       dataset=np.vstack((datapos,negdata))
       
    randindices=np.random.randint(0,dataset.shape[0],obspool_size+testpool_size)
    obspool_ind=randindices[0:obspool_size]
    testpool_ind=randindices[obspool_size:obspool_size+testpool_size]
    return obspool_ind,testpool_ind
    
def process_dataset_pool(obspool_ind,testpool_ind,obs_size,unobs_size,test_size,obsrvd_data_full,unobsrvd_data_full,test_data,unobsrvd_data,path,data_name,obs_feat_num,pos_to_neg):  
    
    filename=path+data_name+'.csv'
    dataset = loadcsv(filename)
    """This is the sampling step in case someone wants to sample"""
    if (pos_to_neg !=0):
       label_indx=dataset.shape[1]-1
       datapos=dataset[dataset[:,label_indx]==1]
       dataneg=dataset[dataset[:,label_indx]==0]
       posnum=datapos.shape[0]
       negnum=posnum* pos_to_neg
       randindices = np.random.randint(0,dataneg.shape[0],negnum)
       print "The length is", len(randindices)
       negdata=dataneg[randindices,0:dataneg.shape[1]]
       dataset=np.vstack((datapos,negdata))
       
    obs,unobs,test = splitdataset_pool(obspool_ind,testpool_ind,dataset,obs_size,test_size)
    #obs,unobs,test = splitdataset(dataset,obs_size,unobs_size,test_size)
    start=0
    end=obs.shape[1]
    write_csv(obsrvd_data_full,obs,start,end)
    write_csv(unobsrvd_data_full,unobs,start,end)
    end=test.shape[1]
    write_csv(test_data,test,start,end)
    end=obs_feat_num
    write_csv(unobsrvd_data,unobs,start,end)
    
def process_dataset(obs_size,unobs_size,test_size,obsrvd_data_full,unobsrvd_data_full,test_data,unobsrvd_data,path,data_name,obs_feat_num,pos_to_neg):  
    
    filename=path+data_name+'.csv'
    dataset = loadcsv(filename)
    """This is the sampling step in case someone wants to sample"""
    if (pos_to_neg !=0):
       label_indx=dataset.shape[1]-1

       datapos=dataset[dataset[:,label_indx]==1]
       dataneg=dataset[dataset[:,label_indx]==0]
       posnum=datapos.shape[0]
       negnum=posnum* pos_to_neg
       randindices = np.random.randint(0,dataneg.shape[0],negnum)
       """For experimentation purpose"""
       #randindices = range(0,negnum)
       negdata=dataneg[randindices,0:dataneg.shape[1]]
       dataset=np.vstack((datapos,negdata))
       #"""Just random reschuffling of the data points"""
       #dataset=np.take(dataset,np.random.permutation(dataset.shape[0]),axis=0,out=dataset)
    
    print "I am in split dataset"
    obs,unobs,test = splitdataset_new(dataset,obs_size,test_size)
    #label_indx=obs.shape[1]-1
    #print "no. of postives in obs",obs[obs[:,label_indx]==1].shape
    #print "no. of postives in unobs",unobs[unobs[:,label_indx]==1].shape
    #obs,unobs,test = splitdataset(dataset,obs_size,unobs_size,test_size)
    start=0
    end=obs.shape[1]
    write_csv(obsrvd_data_full,obs,start,end)
    write_csv(unobsrvd_data_full,unobs,start,end)
    end=test.shape[1]
    write_csv(test_data,test,start,end)
    end=obs_feat_num
    write_csv(unobsrvd_data,unobs,start,end) 
    
def process_dataset2(obs_size,unobs_size,test_size,obsrvd_data_full,unobsrvd_data_full,test_data,unobsrvd_data,path,data_name,obs_feat_num,pos_to_neg):  
    
    filename=path+data_name+'.csv'
    dataset = loadcsv(filename)
    test = splitdataset_fold(dataset,obs_size,test_size)
    #label_indx=obs.shape[1]-1
    #print "no. of postives in obs",obs[obs[:,label_indx]==1].shape
    #print "no. of postives in unobs",unobs[unobs[:,label_indx]==1].shape
    #obs,unobs,test = splitdataset(dataset,obs_size,unobs_size,test_size)
    start=0
    end=obs.shape[1]
    write_csv(obsrvd_data_full,obs,start,end)
    write_csv(unobsrvd_data_full,unobs,start,end)
    end=test.shape[1]
    write_csv(test_data,test,start,end)
    end=obs_feat_num
    write_csv(unobsrvd_data,unobs,start,end) 

"""This is a duplicate function to check certain functionalities"""    
def process_dataset1(obs_size, unobs_size, test_size,obsrvd_data_full,unobsrvd_data_full,test_data,unobsrvd_data,path,data_name,obs_feat_num,pos_to_neg,obs_data_indx,unobs_data_ranking,test_data_indx,unobs_data_idx):  
    
    filename=path+data_name+'.csv'
    dataset = loadcsv(filename)
    """This is the sampling step in case someone wants to sample"""
    if (pos_to_neg !=0):
       label_indx=dataset.shape[1]-1
       datapos=dataset[dataset[:,label_indx]==1]
       dataneg=dataset[dataset[:,label_indx]==0]
       posnum=datapos.shape[0]
       negnum=posnum* pos_to_neg
       randindices = np.random.randint(0,dataneg.shape[0],negnum)
       negdata=dataneg[randindices,0:dataneg.shape[1]]
       dataset=np.vstack((datapos,negdata))
       
    obs,unobs,test = splitdataset1(dataset,obs_size,unobs_size,test_size,path,obs_data_indx,unobs_data_ranking,test_data_indx,unobs_data_idx)
    start=0
    end=obs.shape[1]
    write_csv(obsrvd_data_full,obs,start,end)
    write_csv(unobsrvd_data_full,unobs,start,end)
    end=test.shape[1]
    write_csv(test_data,test,start,end)
    end=obs_feat_num
    write_csv(unobsrvd_data,unobs,start,end) 
    
if __name__ == '__main__':
   
   process_dataset1(obs_size, unobs_size, test_size,obsrvd_data_full,unobsrvd_data_full,test_data,unobsrvd_data,path,data_name,obs_feat_num,pos_to_neg)
            
    