# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:02:10 2017

@author: Srijita
"""
from __future__ import division # floating point division
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from numpy import linalg as LA
import matplotlib.pyplot as plt
import csv
import numpy as np
import Classifier as lmodel
import math
import itertools
import split_data as split
import sys 

"""Efforts were taken to make the code more efficient"""
"""Model on the Observed features of all data built just once"""
#path="D:\\Grad Studies\\SRL\\Active_Feature_acquisition_IJCAI_journal_exp\\Data\\"
path=sys.argv[1]
print "The path is", path
path=path+"\\"+sys.argv[2]+"\\"

""" This name changes according to the dataset name"""
data_name=sys.argv[3]
classifier=sys.argv[4]
metric=sys.argv[5]

"""File names that are not fold dependent"""
clean_data         =  path+data_name+".csv"
test_data          =  path+data_name+"_test.csv"
obs_unobs_all      =  path+data_name+"_obs_unobs_all.csv"


"""Number of features that are observed in both the dataset"""
"""This is one more than the no. of observed features due to the pid introduced"""
obs_feat_num=(int)(sys.argv[6])
#threshold=1
subset_size=(int)(sys.argv[7])

"""These are the parameters to be set by the user"""
no_of_iter_per_run=(int)(sys.argv[8])

no_of_runs=(int)(sys.argv[9])

"""The dataset size needs to be mentioned"""
obs_size=(int)(sys.argv[10])

"""This is not currently required"""
#unobs_size=800
test_split=(int)(sys.argv[11])
#test_size=0.3*obs_size
obs_all_depth=(int)(sys.argv[12])
obs_unobs_depth=(int)(sys.argv[13])
pos_to_neg=(int)(sys.argv[14])

def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset
    
def return_train_label(data,start,end):
    
    train= data[:,start:end]
    label= data[:,data.shape[1]-1]
       
    return train,label
    
def return_train_subset(data,start,end):
    
    train= data[:,start:end]
    label= data[:,data.shape[1]-1]
    label = np.reshape(label,(-1,1))
    train_all=np.hstack((train,label))
       
    return train_all

"""This function learns a model on the entire observed data"""    
def build_learning_model(obs_train,obs_label,depth,classifier):
    learner=lmodel.Logit(depth,classifier)
    learner.learn(obs_train,obs_label)
    return learner
    
def calculate_div(y_obs_all,y_unobs,obs_label,unobs_label,metric):
    
    div_matrix=[]
    div_aggr=[]
    """This loop is for each unobserved points"""
    for i in range(0,len(y_unobs)):
        div_each_pnt=[]
        """
        This loop is for Observed points"""
        for j in range(0,len(y_obs_all)):
               
            """ Some bad approximations for now"""
            if y_obs_all[j][0]==0:
               y_obs_all[j][0]=0.01
            if y_obs_all[j][1] ==0:
               y_obs_all[j][1]=0.01
               
            if y_unobs[i][0]==0:
               y_unobs[i][0]=0.01
            if y_unobs[i][1] ==0:
               y_unobs[i][1]=0.01    
               
            """Calculates divergence based on the parameters"""
            
            #print 'y_unobs[i][0],y_obs_all[j][0],y_unobs[i][1],y_obs_all[j][1]',y_unobs[i][0],y_obs_all[j][0],y_unobs[i][1],y_obs_all[j][1]
            if metric=="KL":
               div=y_unobs[i][0]*math.log((y_unobs[i][0]/y_obs_all[j][0]),2)+ y_unobs[i][1]*math.log((y_unobs[i][1]/y_obs_all[j][1]),2)
            elif metric=="HD":
               div=(1/math.sqrt(2))*(math.sqrt((math.sqrt(y_unobs[i][0])-math.sqrt(y_obs_all[j][0]))**2  + (math.sqrt(y_unobs[i][1])-math.sqrt(y_obs_all[j][1]))**2))
            elif metric=="TV":
               div= (1/2)*(math.fabs(y_unobs[i][0]- y_obs_all[j][0]) + math.fabs(y_unobs[i][1]- y_obs_all[j][1]))
            elif metric=="CD":    
               #print "I am calculating CD" 
               div= ((y_unobs[i][0]- y_obs_all[j][0])**2/y_obs_all[j][0]) + ((y_unobs[i][1]- y_obs_all[j][1])**2/y_obs_all[j][1])
                  
            div_each_pnt.append(div)  
        #div_each_pnt=np.asarray(div_each_pnt)
        div_mean=np.average(div_each_pnt)
        #div_mean=np.amax(div_each_pnt)
        #div_mean=np.amin(div_each_pnt)
        div_matrix.append(div_each_pnt)
        div_aggr.append(div_mean)
    return div_matrix,div_aggr 
       
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    #return (correct/float(len(ytest))) * 100.0   
    return (correct/float(len(ytest)))

"""This is the main algorithm flow"""

def process_algo(obsdata,unobsdata,unobsdata_all,learner,learner_all_feat):
   
   print "The shape is",obsdata.shape,unobsdata.shape,unobsdata_all.shape,testdata.shape
   label_indx=obsdata.shape[1]-1
   obspos=obsdata[obsdata[:,label_indx]==1].shape[0]
   obsneg=obsdata[obsdata[:,label_indx]==0].shape[0]
   print "The class distribution is", obspos/(obspos+obsneg),obsneg/(obspos+obsneg)
           
   """Converting Obs data to train and label"""
   start=1;end=obs_feat_num
   obs_train,obs_label=return_train_label(obsdata,start,end)
   
   """Stacking the observed data and observed features of the unobserved data together"""
   start=1;end=obs_feat_num
   obsdata_some=return_train_subset(obsdata,start,end)
   unobsdata_clean=unobsdata[:,1:unobsdata.shape[1]]
   obs_unobs_data=np.vstack((obsdata_some,unobsdata_clean))
   start=0;end=obs_feat_num-1
   obs_unobs_train,obs_unobs_label=return_train_label(obs_unobs_data,start,end)
   
   """Converting Unobs data to train and label"""
   start=1;end=unobsdata.shape[1]-1
   unobs_train,unobs_label=return_train_label(unobsdata,start,end)
   
   """Learn the model from observed features of the entire data pool  ---->>   STEP-1"""
   #obs_unobs_label=obs_unobs_label[obs_unobs_label[:,0]==-1]
   #learner=build_learning_model(obs_unobs_train,obs_unobs_label,obs_unobs_depth,classifier)  
   
   """Doing predictions on Observed training data"""
   #y_obs=learner.predict_prob(obs_train)
    
   """Doing predictions on Observed features of unobserved data"""
   y_unobs=learner.predict_prob(unobs_train)
   #print 'yobs is',y_obs
   #print 'yunobs is', y_unobs   
   """Learn a model on all the features of the observed dataset"""
   start=1;end=obsdata.shape[1]-1
   obs_all_train,obs_all_label=return_train_label(obsdata,start,end)
   
   """This is added for efficiency; so that model on all the features are not build twice in the code"""
   if learner_all_feat==0:
      print "I am building the Initial model on all the features of Observed set" 
      learner_all_feat=build_learning_model(obs_all_train,obs_all_label,obs_all_depth,classifier)  
   y_obs_all=learner_all_feat.predict_prob(obs_all_train)
   
   """Calculation of KL Divergence between conditionals of each data point
      in the unobserved set and all points of the observed set----->>  STEP2"""
   
   
   div_matrix,div_aggr = calculate_div(y_obs_all,y_unobs,obs_label,unobs_label,metric)  
   div_matrix=np.asarray(div_matrix)
   div_aggr=np.asarray(div_aggr)
   div_aggr_sorted = np.sort(div_aggr)[::-1]
   
   """Converting the KL-div matix into a distribution"""
#   div_prob=[]
#   div_sum=np.sum(div_aggr)
#   for i in range(0,div_aggr.shape[0]):
#       div_prob.append(div_aggr[i]/div_sum)
#   div_prob=np.asarray(div_prob)
#   unobs_range=range(0,div_prob.shape[0])
#   unobs_indx=np.random.choice(unobs_range,subset_size, p=div_prob)
   #print "The sorted divergence array is", div_aggr_sorted[0:10]
   """Rank the points in descending order acc to their distance from the mean distribution of Observed points"""
   
   unobs_indx=np.argsort(div_aggr)[::-1][:len(div_aggr)]
   
   return unobs_indx

"""This model can be reused for each iteration of act learning for building the model on the whole of observed data"""
def build_model_on_all_data_obs_feat(obsdata,unobsdata,obs_unobs_depth,classifier):
    
   """Converting Obs data to train and label"""
   start=1;end=obs_feat_num
   obs_train,obs_label=return_train_label(obsdata,start,end)
   
   """Stacking the observed data and observed features of the unobserved data together"""
   start=1;end=obs_feat_num
   obsdata_some=return_train_subset(obsdata,start,end)
   unobsdata_clean=unobsdata[:,1:unobsdata.shape[1]]
   obs_unobs_data=np.vstack((obsdata_some,unobsdata_clean))
   start=0;end=obs_feat_num-1
   obs_unobs_train,obs_unobs_label=return_train_label(obs_unobs_data,start,end)
   learner=build_learning_model(obs_unobs_train,obs_unobs_label,obs_unobs_depth,classifier)  
   return learner
           
if __name__ == '__main__':
   
   #sys.stdout = open(path+'log.txt', 'w')
   np.set_printoptions(threshold='nan')
   
   """Data structure for all the statistics"""
   acc_sugg_iter= [0]*no_of_iter_per_run
   acc_rnd_iter=  [0]*no_of_iter_per_run
   acc_std_dev_sugg=[0]*no_of_iter_per_run
   acc_std_dev_rnd=[0]*no_of_iter_per_run
   
   roc_sugg_iter= [0]*no_of_iter_per_run
   roc_rnd_iter=  [0]*no_of_iter_per_run
   #roc_std_dev_sugg=[0]*no_of_iter_per_run
   #roc_std_dev_rnd=[0]*no_of_iter_per_run
   
   pr_sugg_iter=  [0]*no_of_iter_per_run
   pr_rnd_iter=   [0]*no_of_iter_per_run
   #pr_std_dev_sugg=[0]*no_of_iter_per_run
   #pr_std_dev_rnd=[0]*no_of_iter_per_run
   
   recall_sugg_iter=  [0]*no_of_iter_per_run
   recall_rnd_iter=   [0]*no_of_iter_per_run
   recall_std_dev_sugg=[0]*no_of_iter_per_run
   recall_std_dev_rnd=[0]*no_of_iter_per_run
   
   f1_sugg_iter=  [0]*no_of_iter_per_run
   f1_rnd_iter=   [0]*no_of_iter_per_run
   f1_std_dev_sugg=[0]*no_of_iter_per_run
   f1_std_dev_rnd=[0]*no_of_iter_per_run
   
   f3_sugg_iter=  [0]*no_of_iter_per_run
   f3_rnd_iter=   [0]*no_of_iter_per_run
   
   f5_sugg_iter=  [0]*no_of_iter_per_run
   f5_rnd_iter=   [0]*no_of_iter_per_run
   f5_std_dev_sugg=[0]*no_of_iter_per_run
   f5_std_dev_rnd=[0]*no_of_iter_per_run
   
   
   """Data structures to calculate value across each run"""
   acc_all_run_data_sugg=[]
   recall_all_run_data_sugg=[]
   f1_all_run_data_sugg=[]
   f5_all_run_data_sugg=[]
   
   acc_all_run_data_rnd=[]
   recall_all_run_data_rnd=[]
   f1_all_run_data_rnd=[]
   f5_all_run_data_rnd=[]
      
   first=0
   for runs in range(0,no_of_runs): 
       
       """Capture the performance metric for each single run"""
       acc_each_run_sugg=[]
       recall_each_run_sugg=[]
       f1_each_run_sugg=[]
       f5_each_run_sugg=[]
       
       acc_each_run_rnd=[]
       recall_each_run_rnd=[]
       f1_each_run_rnd=[]
       f5_each_run_rnd=[]

       """This step calls the program to create the observed and unobserved dataset from the total dataset"""
       print "*********RUN NO***********", runs
       
        
       """Define the folder names according to the classifier and the divergences"""
       """Classifier"""
       if classifier=="GB":
            cfname="Gradient_Boosting"
       elif classifier=="LR":
            cfname="Logistic_Regression"
       elif classifier=="SVM":
            cfname="SVM"
       elif classifier=="DT":
            cfname="Decision_Tree"
       
       """Divergence metric""" 
       if   metric=="KL":
            mfname="KL_div"
       elif metric=="HD":
            mfname="Hell_dist"
       elif metric=="TV":
            mfname="Total_var"
       elif metric=="CD":
            mfname="Chi"     
       
       """This is the graph path after adding the classifier and divergence""" 
       gpath=path+"\\Graphs\\"+cfname+"\\"+mfname+"\\"
       
       """Basic statistics and standard deviation file name"""
       stat_data=path+"\\All_runs\\"+cfname+"\\"+mfname+"\\"+data_name+"_stat.csv"
       std_dev_data=path+"\\All_runs\\"+cfname+"\\"+mfname+"\\"+data_name+"_std_dev.csv"
       
       """Run details file name"""
       acc_file_sugg=path+"\\All_runs\\"+cfname+"\\"+mfname+"\\"+"Acc_sugg.csv"
       recall_file_sugg=path+"\\All_runs\\"+cfname+"\\"+mfname+"\\"+"Recall_sugg.csv"
       F1_file_sugg=path+"\\All_runs\\"+cfname+"\\"+mfname+"\\"+"F1_sugg.csv"
       gmean_file_sugg=path+"\\All_runs\\"+cfname+"\\"+mfname+"\\"+"gmean_sugg.csv"
       
       acc_file_rnd=path+"\\All_runs\\"+cfname+"\\"+mfname+"\\"+"Acc_rnd.csv"
       recall_file_rnd=path+"\\All_runs\\"+cfname+"\\"+mfname+"\\"+"Recall_rnd.csv"
       F1_file_rnd=path+"\\All_runs\\"+cfname+"\\"+mfname+"\\"+"F1_rnd.csv"
       gmean_file_rnd=path+"\\All_runs\\"+cfname+"\\"+mfname+"\\"+"gmean_rnd.csv"
       
       fpath=path+"Fold"+str(runs+1)+"\\"
       
       """File names that are not fold dependent"""
       obsrvd_data_full   =  fpath+data_name+"_obs.csv"
       unobsrvd_data_full =  fpath+data_name+"_unobs_all.csv" 
       unobsrvd_data      =  fpath +data_name+"_unobs.csv"
       acquired_data      =  fpath+data_name+"acquired.csv"
       
       #split.process_dataset_act(obs_size,test_split,obsrvd_data_full,unobsrvd_data_full,test_data,unobsrvd_data,obs_unobs_all,path,data_name,obs_feat_num,pos_to_neg,first)
       first=1
       #split.process_dataset_act(obs_size,test_size,obsrvd_data_full,unobsrvd_data_full,test_data,unobsrvd_data,path,data_name,obs_feat_num,pos_to_neg)

       """Load all the datasets into the program"""
       alldata   =     loadcsv(clean_data)
       obsdata   =     loadcsv(obsrvd_data_full)
       unobsdata =     loadcsv(unobsrvd_data)
       unobsdata_all=  loadcsv(unobsrvd_data_full)
       testdata=       loadcsv(test_data)
       
       learner=build_model_on_all_data_obs_feat(obsdata,unobsdata,obs_unobs_depth,classifier)
       #tprint 'testdata is', testdata[0]
       learner_tot_sug=0
       for i in range(0,no_of_iter_per_run):
           
           unobs_indx=process_algo(obsdata,unobsdata,unobsdata_all,learner,learner_tot_sug)
           
           
           """*************************************************************************************************************"""
           print "The Evaluation phase is now beginning"
           """Evaluation part of the algorithm"""
           
           """Test data formation"""
           start=1;end=testdata.shape[1]-1
           testdata_train,testdata_label=return_train_label(testdata,start,end)
           
           """Evaluating on test data by learning on full obs and suggested unobs"""
           
           unobs_indx_final=unobs_indx[0:subset_size]
           unobs_sug=unobsdata_all[unobs_indx_final[0:subset_size],0:unobsdata_all.shape[1]]
           totdata_sug=np.vstack((obsdata,unobs_sug))
           
           end=totdata_sug.shape[1]-1
           totdata_sug_train,totdata_sug_label=return_train_label(totdata_sug,start,end)
           learner_tot_sug=build_learning_model(totdata_sug_train,totdata_sug_label,obs_all_depth,classifier) 
           y_tot_sug=learner_tot_sug.predict(testdata_train)
           acc_sugg=getaccuracy(testdata_label,y_tot_sug)
           acc_each_run_sugg.append(acc_sugg)
           acc_sugg_iter[i]=acc_sugg_iter[i]+acc_sugg
           roc_sugg=roc_auc_score(testdata_label, y_tot_sug)
           roc_sugg_iter[i]=roc_sugg_iter[i]+roc_sugg
           pr_sugg=average_precision_score(testdata_label, y_tot_sug)
           pr_sugg_iter[i]=pr_sugg_iter[i]+pr_sugg
           recall_sugg=recall_score(testdata_label, y_tot_sug)
           recall_each_run_sugg.append(recall_sugg)
           print "Recall suggested  is", recall_sugg
           recall_sugg_iter[i]=recall_sugg_iter[i]+recall_sugg
           f1_sugg=f1_score(testdata_label, y_tot_sug)
           f1_each_run_sugg.append(f1_sugg)
           f1_sugg_iter[i]=f1_sugg_iter[i]+f1_sugg
           f3_sugg=fbeta_score(testdata_label, y_tot_sug,beta=3)
           f3_sugg_iter[i]=f3_sugg_iter[i]+f3_sugg
           f5_sugg=geometric_mean_score(testdata_label, y_tot_sug)
           f5_each_run_sugg.append(f5_sugg)
           f5_sugg_iter[i]=f5_sugg_iter[i]+f5_sugg
           
           
           
           """Evaluating on test data by learning on full obs and random unobs"""
           
           if (i==0):
               obsdata_rnd=obsdata
               unobsdata_all_rnd=unobsdata_all
               
           randindices = np.random.randint(0,unobsdata_all_rnd.shape[0],subset_size)
           #print randindices
           #print "The shape of unobsdata_all_rnd.shape[0]",unobsdata_all_rnd.shape[0]
           #print "unobs_data",unobsdata_all[unobs]
           unobs_random=unobsdata_all_rnd[randindices,0:unobsdata_all_rnd.shape[1]]
           #print "The shape of obsdata_rnd is", obsdata_rnd.shape
           #print "The shape of unobsdata_all_rnd is", unobsdata_all_rnd.shape
           totdata_rnd=np.vstack((obsdata_rnd,unobs_random))
           #print "The shape of totdata_rnd is", totdata_rnd.shape
           start=1;end=totdata_rnd.shape[1]-1
           totdata_rnd_train,totdata_rnd_label=return_train_label(totdata_rnd,start,end)
           learner_tot_rnd=build_learning_model(totdata_rnd_train,totdata_rnd_label,obs_all_depth,classifier) 
           y_tot_rnd=learner_tot_rnd.predict(testdata_train)
           acc_rnd=getaccuracy(testdata_label,y_tot_rnd)
           acc_each_run_rnd.append(acc_rnd)
           acc_rnd_iter[i]=acc_rnd_iter[i]+acc_rnd
           roc_rnd=roc_auc_score(testdata_label, y_tot_rnd)
           roc_rnd_iter[i]=roc_rnd_iter[i]+roc_rnd
           pr_rnd=average_precision_score(testdata_label, y_tot_rnd)
           pr_rnd_iter[i]=pr_rnd_iter[i]+pr_rnd
           recall_rnd=recall_score(testdata_label, y_tot_rnd)
           print "Recall random  is", recall_rnd
           recall_each_run_rnd.append(recall_rnd)
           recall_rnd_iter[i]=recall_rnd_iter[i]+recall_rnd
           f1_rnd=f1_score(testdata_label, y_tot_rnd)
           f1_each_run_rnd.append(f1_rnd)
           f1_rnd_iter[i]=f1_rnd_iter[i]+f1_rnd
           f3_rnd=fbeta_score(testdata_label, y_tot_rnd,beta=3)
           f3_rnd_iter[i]=f3_rnd_iter[i]+f3_rnd
           f5_rnd=geometric_mean_score(testdata_label, y_tot_rnd)
           f5_each_run_rnd.append(f5_rnd)
           f5_rnd_iter[i]=f5_rnd_iter[i]+f5_rnd
           
           
           """Assigning the correct observed and unobserved data points for next iteration to the random algorithm"""
           obsdata_rnd=totdata_rnd
           unobsdata_all_rnd=np.delete(unobsdata_all_rnd, randindices, axis=0)
                     
           """Assigning the correct observed and unobserved data points for next iteration to the algorithm"""
           obsdata=totdata_sug
           unobsdata=np.delete(unobsdata, unobs_indx_final, axis=0)
           unobsdata_all=np.delete(unobsdata_all, unobs_indx_final, axis=0)
           
           """To see the current distribution"""
           label_indx=obsdata.shape[1]-1
           obspos=obsdata[obsdata[:,label_indx]==1].shape[0]
           obsneg=obsdata[obsdata[:,label_indx]==0].shape[0]
#           if ((obsneg/obspos) > 1.5):
#               #print "The class distribution is", obspos/(obspos+obsneg),obsneg/(obspos+obsneg)
#               print "The iteration number is", i 
#               break
       """Populating the data structures with an entire run detail"""
       acc_all_run_data_sugg.append(acc_each_run_sugg)
       recall_all_run_data_sugg.append(recall_each_run_sugg)
       f1_all_run_data_sugg.append(f1_each_run_sugg)
       f5_all_run_data_sugg.append(f5_each_run_sugg)
       
       acc_all_run_data_rnd.append(acc_each_run_rnd)
       recall_all_run_data_rnd.append(recall_each_run_rnd)
       f1_all_run_data_rnd.append(f1_each_run_rnd)
       f5_all_run_data_rnd.append(f5_each_run_rnd)
       
       """Writing the acquired dataset to csvfile""" 
       #if i== (no_of_iter_per_run-1):
       with open(acquired_data, 'wb') as csvfile1:         
            csvwriter = csv.writer(csvfile1, delimiter=',')
            #csvwriter.writerow(["Acc_sugg","Acc_rnd","ROC_sugg","ROC_rnd","pr_sugg","pr_rnd","recall_sugg","recall_rnd","f1_sugg","f1_rnd","f5_sugg","f5_rnd"])
            for i in range(0,len(obsdata)):
                csvwriter.writerow(obsdata[i])
   
#   """Learning a model on the entire training data"""
#   obs_unobs_data=loadcsv(obs_unobs_all)
#   #print "THe shape is", obs_unobs_data.shape
#   start=1;end=obs_unobs_data.shape[1]-1
#   obs_unobs_all_train,obs_unobs_all_label=return_train_label(obs_unobs_data,start,end)
#   learner_tot_all=build_learning_model(obs_unobs_all_train,obs_unobs_all_label,obs_all_depth) 
#   y_tot_all=learner_tot_all.predict(testdata_train)
#   #print "THe shape is", y_tot_all.shape
#   acc_all=getaccuracy(testdata_label,y_tot_all)
#   roc_all=roc_auc_score(testdata_label, y_tot_all)
#   pr_all=average_precision_score(testdata_label, y_tot_all)
   
   """Calculating the mean accuracies over all the runs for both the algorithms"""
   for i in range(0,len(acc_sugg_iter)):
       acc_sugg_iter[i]=acc_sugg_iter[i]/no_of_runs
       acc_rnd_iter[i]=acc_rnd_iter[i]/no_of_runs
       #acc_all_iter[i]=acc_all
       roc_sugg_iter[i]=roc_sugg_iter[i]/no_of_runs
       roc_rnd_iter[i]=roc_rnd_iter[i]/no_of_runs
       #roc_all_iter[i]=roc_all
       pr_sugg_iter[i]=pr_sugg_iter[i]/no_of_runs
       pr_rnd_iter[i]=pr_rnd_iter[i]/no_of_runs
       #pr_all_iter[i]=pr_all     
       recall_sugg_iter[i]=recall_sugg_iter[i]/no_of_runs
       recall_rnd_iter[i]=recall_rnd_iter[i]/no_of_runs
       
       f1_sugg_iter[i]=f1_sugg_iter[i]/no_of_runs
       f1_rnd_iter[i]=f1_rnd_iter[i]/no_of_runs
       
       f3_sugg_iter[i]=f3_sugg_iter[i]/no_of_runs
       f3_rnd_iter[i]=f3_rnd_iter[i]/no_of_runs
       
       f5_sugg_iter[i]=f5_sugg_iter[i]/no_of_runs
       f5_rnd_iter[i]=f5_rnd_iter[i]/no_of_runs
   
   """Getting the standard deviation""" 
#   acc_mean_sugg=np.mean(acc_sugg_iter)
#   acc_mean_rnd=np.mean(acc_rnd_iter)
#   roc_mean_sugg=np.mean(roc_sugg_iter)
#   roc_mean_rnd=np.mean(roc_rnd_iter)
#   pr_mean_sugg=np.mean(pr_sugg_iter)
#   pr_mean_rnd=np.mean(pr_rnd_iter)
   
   for i in range(0,1):
       #print i
       print "Accuracy,ROC,PR,recall,f1,f5 suggested", acc_sugg_iter[i],roc_sugg_iter[i],pr_sugg_iter[i],recall_sugg_iter[i],f1_sugg_iter[i],f5_sugg_iter[i]
       print "Accuracy ROC,PR,recall,f1,f5 random",    acc_rnd_iter[i],roc_rnd_iter[i],pr_rnd_iter[i],recall_rnd_iter[i],f1_rnd_iter[i],f5_rnd_iter[i]
       
   """Calculate the deviations from mean"""
#   j=19
#   for i in range(0,len(acc_sugg_iter)):
#       if  (acc_std_dev_sugg[i] > acc_rnd_iter[i]):
#           acc_std_dev_sugg[i]=abs(acc_sugg_iter[i]-acc_mean_sugg)
#           #acc_std_dev_rnd[i]=abs(acc_rnd_iter[i]-acc_mean_rnd)
#       if (roc_sugg_iter[i]  > roc_rnd_iter[i]):
#           roc_std_dev_sugg[i]=abs(roc_sugg_iter[i]-roc_mean_sugg)
#           #roc_std_dev_rnd[i]=abs(roc_rnd_iter[i]-roc_mean_rnd)
#       if (pr_sugg_iter[i] > pr_rnd_iter[i]):   
#           pr_std_dev_sugg[i]=abs(pr_sugg_iter[i]-pr_mean_sugg)
#           #pr_std_dev_rnd[i]=abs(pr_rnd_iter[i]-pr_mean_rnd)
   """File writing section"""
   """Populate all run detail in a file for suggested method"""
   with open(acc_file_sugg, 'wb') as csvfile:         
        csvwriter = csv.writer(csvfile, delimiter=',')
        for i in range(0,no_of_runs):
            csvwriter.writerow(acc_all_run_data_sugg[i])
            
   with open(recall_file_sugg, 'wb') as csvfile1:         
        csvwriter = csv.writer(csvfile1, delimiter=',')
        for i in range(0,no_of_runs):
            csvwriter.writerow(recall_all_run_data_sugg[i])  
   
   with open(F1_file_sugg, 'wb') as csvfile2:         
        csvwriter = csv.writer(csvfile2, delimiter=',')
        for i in range(0,no_of_runs):
            csvwriter.writerow(f1_all_run_data_sugg[i]) 
            
   with open(gmean_file_sugg, 'wb') as csvfile3:         
        csvwriter = csv.writer(csvfile3, delimiter=',')
        for i in range(0,no_of_runs):
            csvwriter.writerow(f5_all_run_data_sugg[i]) 
            
   """Populate all run detail in a file for Random baseline"""
   with open(acc_file_rnd, 'wb') as csvfile:         
        csvwriter = csv.writer(csvfile, delimiter=',')
        for i in range(0,no_of_runs):
            csvwriter.writerow(acc_all_run_data_rnd[i])
            
   with open(recall_file_rnd, 'wb') as csvfile1:         
        csvwriter = csv.writer(csvfile1, delimiter=',')
        for i in range(0,no_of_runs):
            csvwriter.writerow(recall_all_run_data_rnd[i])  
   
   with open(F1_file_rnd, 'wb') as csvfile2:         
        csvwriter = csv.writer(csvfile2, delimiter=',')
        for i in range(0,no_of_runs):
            csvwriter.writerow(f1_all_run_data_rnd[i]) 
            
   with open(gmean_file_rnd, 'wb') as csvfile3:         
        csvwriter = csv.writer(csvfile3, delimiter=',')
        for i in range(0,no_of_runs):
            csvwriter.writerow(f5_all_run_data_rnd[i])        
   """FILE WRITING SECTION ENDS"""
   
   """Calculate the standard deviation of the results"""
   acc_all_run_data_sugg1=np.asarray(acc_all_run_data_sugg)
   recall_all_run_data_sugg1=np.asarray(recall_all_run_data_sugg) 
   f1_all_run_data_sugg1=np.asarray(f1_all_run_data_sugg)
   f5_all_run_data_sugg1=np.asarray(f5_all_run_data_sugg)

   acc_all_run_data_rnd1=np.asarray(acc_all_run_data_rnd)
   recall_all_run_data_rnd1=np.asarray(recall_all_run_data_rnd) 
   f1_all_run_data_rnd1=np.asarray(f1_all_run_data_rnd)
   f5_all_run_data_rnd1=np.asarray(f5_all_run_data_rnd)     
         
   for i in range(0,no_of_iter_per_run):
       #print "The shape is",recall_all_run_data_sugg1[:,i].shape
       print "The std dev recall is", np.std(recall_all_run_data_sugg1[:,i])
       acc_std_dev_sugg[i]=np.std(acc_all_run_data_sugg1[:,i])
       recall_std_dev_sugg[i]=np.std(recall_all_run_data_sugg1[:,i])
       f1_std_dev_sugg[i]=np.std(f1_all_run_data_sugg1[:,i])
       f5_std_dev_sugg[i]=np.std(f5_all_run_data_sugg1[:,i])
       
       acc_std_dev_rnd[i]=np.std(acc_all_run_data_rnd1[:,i])
       recall_std_dev_rnd[i]=np.std(recall_all_run_data_rnd1[:,i])
       f1_std_dev_rnd[i]=np.std(f1_all_run_data_rnd1[:,i])
       f5_std_dev_rnd[i]=np.std(f5_all_run_data_rnd1[:,i])
#              
   """Statistics summary plotting""" 
   """Plot the accuracies of both the algorithms"""
   acq_size = range(0,no_of_iter_per_run)
   """Blue color is for suggested algorithm accuracy"""
   plt.plot(acq_size,acc_sugg_iter,color='b')
   #plt.errorbar(acq_size,acc_sugg_iter,yerr=acc_std_dev_sugg,color='b',capsize=5,markeredgewidth=2 )
   """Red color is for random algorithm accuracy"""
   plt.plot(acq_size,acc_rnd_iter,color='r')
   #plt.errorbar(acq_size,acc_rnd_iter,yerr=acc_std_dev_rnd,color='r',capsize=5,markeredgewidth=2 )
   #"""Magenta color is for all data accuracy"""
   #plt.plot(acq_size,acc_all_iter,color='m')
   plt.xlim(0,no_of_iter_per_run)
   plt.ylim(0,1)
   plt.xlabel('No of acquired instances')
   plt.ylabel('Accuracy')
   plt.savefig(gpath+'Accuracy.png', bbox_inches='tight')   
   
   plt.clf()
   """Plot the roc of both the algorithms"""
   #acq_size = range(0,no_of_iter_per_run)
   """Blue color is for suggested algorithm accuracy"""
   plt.plot(acq_size,roc_sugg_iter,color='b')
   #plt.errorbar(acq_size,roc_sugg_iter,yerr=roc_std_dev_sugg,color='b',capsize=5,markeredgewidth=2 )
   """Red color is for random algorithm accuracy"""
   plt.plot(acq_size,roc_rnd_iter,color='r')
   #plt.errorbar(acq_size,roc_rnd_iter,yerr=roc_std_dev_rnd,color='r',capsize=5,markeredgewidth=2 )
   #"""Magenta color is for all data accuracy"""
   #plt.plot(acq_size,roc_all_iter,color='m')
   plt.xlim(0,no_of_iter_per_run)
   plt.ylim(0,1)
   plt.xlabel('No of acquired instances')
   plt.ylabel('ROC')
   plt.savefig(gpath+'ROC.png', bbox_inches='tight')
   
   plt.clf()
   """Plot the pr of both the algorithms"""
   #acq_size = range(0,no_of_iter_per_run)
   """Blue color is for suggested algorithm accuracy"""
   plt.plot(acq_size,pr_sugg_iter,color='b')
   #plt.errorbar(acq_size,pr_sugg_iter,yerr=pr_std_dev_sugg,color='b',capsize=5,markeredgewidth=2 )
   """Red color is for random algorithm accuracy"""
   plt.plot(acq_size,pr_rnd_iter,color='r')
   #plt.errorbar(acq_size,pr_rnd_iter,yerr=pr_std_dev_rnd,color='r',capsize=5,markeredgewidth=2 )
   #"""Magenta color is for all data accuracy"""
   #plt.plot(acq_size,acc_all_iter,color='m')
   plt.xlim(0,no_of_iter_per_run)
   plt.ylim(0,1)
   plt.xlabel('No of acquired instances')
   plt.ylabel('PR')
   plt.savefig(gpath+'PR.png', bbox_inches='tight')  
   
   plt.clf()
   """Plot the recall of both the algorithms"""
   #acq_size = range(0,no_of_iter_per_run)
   """Blue color is for suggested algorithm accuracy"""
   plt.plot(acq_size,recall_sugg_iter,color='b')
   #plt.errorbar(acq_size,pr_sugg_iter,yerr=pr_std_dev_sugg,color='b',capsize=5,markeredgewidth=2 )
   """Red color is for random algorithm accuracy"""
   plt.plot(acq_size,recall_rnd_iter,color='r')
   #plt.errorbar(acq_size,pr_rnd_iter,yerr=pr_std_dev_rnd,color='r',capsize=5,markeredgewidth=2 )
   #"""Magenta color is for all data accuracy"""
   #plt.plot(acq_size,acc_all_iter,color='m')
   plt.xlim(0,no_of_iter_per_run)
   plt.ylim(0,1)
   plt.xlabel('No of acquired instances')
   plt.ylabel('Recall')
   plt.savefig(gpath+'Recall.png', bbox_inches='tight') 
   
   plt.clf()
   """Plot the f1 of both the algorithms"""
   #acq_size = range(0,no_of_iter_per_run)
   """Blue color is for suggested algorithm accuracy"""
   plt.plot(acq_size,f1_sugg_iter,color='b')
   #plt.errorbar(acq_size,pr_sugg_iter,yerr=pr_std_dev_sugg,color='b',capsize=5,markeredgewidth=2 )
   """Red color is for random algorithm accuracy"""
   plt.plot(acq_size,f1_rnd_iter,color='r')
   #plt.errorbar(acq_size,pr_rnd_iter,yerr=pr_std_dev_rnd,color='r',capsize=5,markeredgewidth=2 )
   #"""Magenta color is for all data accuracy"""
   #plt.plot(acq_size,acc_all_iter,color='m')
   plt.xlim(0,no_of_iter_per_run)
   plt.ylim(0,1)
   plt.xlabel('No of acquired instances')
   plt.ylabel('f1')
   plt.savefig(gpath+'F1.png', bbox_inches='tight') 
   
   plt.clf()
   """Plot the f5 of both the algorithms"""
   #acq_size = range(0,no_of_iter_per_run)
   """Blue color is for suggested algorithm accuracy"""
   plt.plot(acq_size,f3_sugg_iter,color='b')
   #plt.errorbar(acq_size,pr_sugg_iter,yerr=pr_std_dev_sugg,color='b',capsize=5,markeredgewidth=2 )
   """Red color is for random algorithm accuracy"""
   plt.plot(acq_size,f3_rnd_iter,color='r')
   #plt.errorbar(acq_size,pr_rnd_iter,yerr=pr_std_dev_rnd,color='r',capsize=5,markeredgewidth=2 )
   #"""Magenta color is for all data accuracy"""
   #plt.plot(acq_size,acc_all_iter,color='m')
   plt.xlim(0,no_of_iter_per_run)
   plt.ylim(0,1)
   plt.xlabel('No of acquired instances')
   plt.ylabel('f3')
   plt.savefig(gpath+'F3.png', bbox_inches='tight') 
   
   plt.clf()
   """Plot the f5 of both the algorithms"""
   #acq_size = range(0,no_of_iter_per_run)
   """Blue color is for suggested algorithm accuracy"""
   plt.plot(acq_size,f5_sugg_iter,color='b')
   #plt.errorbar(acq_size,pr_sugg_iter,yerr=pr_std_dev_sugg,color='b',capsize=5,markeredgewidth=2 )
   """Red color is for random algorithm accuracy"""
   plt.plot(acq_size,f5_rnd_iter,color='r')
   #plt.errorbar(acq_size,pr_rnd_iter,yerr=pr_std_dev_rnd,color='r',capsize=5,markeredgewidth=2 )
   #"""Magenta color is for all data accuracy"""
   #plt.plot(acq_size,acc_all_iter,color='m')
   plt.xlim(0,no_of_iter_per_run)
   plt.ylim(0,1)
   plt.xlabel('No of acquired instances')
   plt.ylabel('gmean')
   plt.savefig(gpath+'gmean.png', bbox_inches='tight') 
   
   """Writing all the statistics to csv Files"""
   with open(stat_data, 'wb') as csvfile:         
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(["Acc_sugg","Acc_rnd","ROC_sugg","ROC_rnd","pr_sugg","pr_rnd","recall_sugg","recall_rnd","f1_sugg","f1_rnd","gmean_sugg","gmean_rnd","f3_sugg","f3_rnd"])
        for i in range(0,len(acc_sugg_iter)):
            csvwriter.writerow([acc_sugg_iter[i],acc_rnd_iter[i],roc_sugg_iter[i],roc_rnd_iter[i],pr_sugg_iter[i],pr_rnd_iter[i],recall_sugg_iter[i],recall_rnd_iter[i],f1_sugg_iter[i],f1_rnd_iter[i],f5_sugg_iter[i],f5_rnd_iter[i],f3_sugg_iter[i],f3_rnd_iter[i]])
   
   """Writing all the std dev to csv Files"""
   with open(std_dev_data, 'wb') as csvfile:         
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(["Acc_sugg","Acc_rnd","recall_sugg","recall_rnd","f1_sugg","f1_rnd","gmean_sugg","gmean_rnd"])
        for i in range(0,len(acc_std_dev_sugg)):
            csvwriter.writerow([acc_std_dev_sugg[i],acc_std_dev_rnd[i],recall_std_dev_sugg[i],recall_std_dev_rnd[i],f1_std_dev_sugg[i],f1_std_dev_rnd[i],f5_std_dev_sugg[i],f5_std_dev_rnd[i]])         
   