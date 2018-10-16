# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:25:17 2017

@author: Srijita
"""
from __future__ import division # floating point division
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
#from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import numpy as np

class Logit:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects value between max and min in training set.
    """
    
    def __init__( self,depth,classifier):
        """ Params can contain any useful parameters for the algorithm; the weights are learnt """
        
        """XGBOOST package had to be installed separately into ANACONDA"""
        #self.model=XGBClassifier(objective='binary:logitraw',subsample=0.8,scale_pos_weight=ratio,learning_rate=0.1,max_delta_step=1)
        #self.model=XGBClassifier(max_depth=depth)
        if   classifier=="LR":
             self.model=LogisticRegression()
        elif classifier=="DT": 
             self.model=DecisionTreeClassifier(max_depth=depth)
        elif classifier=="SVM":   
             #self.model = SVC(kernel='rbf',probability=1)
             #self.model = SVC(kernel='poly',probability=1,degree=2)
             self.model = SVC(kernel='linear',probability=1)
        elif classifier=="GB":     
             self.model=GradientBoostingClassifier(max_depth=depth)
        
    def learn(self, Xdata, ydata):
        """ Learns using the traindata """
        
        #self.model.fit(Xdata,ydata,eval_metric=['auc','ams@0.15'])
        #self.model.fit(Xdata,ydata,eval_metric='auc')
        print "The shape of Xdata and ydata are", Xdata.shape, ydata.shape
        #print Xdata[0]
        #print ydata[0:10]
        self.model.fit(Xdata,ydata)
        #print self.model.coef_
        #print "Random Forest Classifier built"
        #print self.model.tree_.value

    def predict_prob(self, Xdata):
        """predicts probability of Test Data"""
        ydata=self.model.predict_proba(Xdata)
        return ydata    
        
    def predict(self, Xdata):
        """predicts label of Test Data"""
        ydata=self.model.predict(Xdata)
        return ydata 
    
#    def get_coeff(self):
#        """give the weights of the features"""
#        print "The weights are",self.model.coef_
#        return self.model.coef_.ravel()
        
class Kmeans:
    
      """Implement the learning and prediction for K-means algorithm"""
      
      def __init__( self, params=None ):
          self.model=KMeans(n_clusters=params, random_state=0)
          
      
      def learn(self, Xdata):
          self.model.fit(Xdata)
          print 'Doing K-means Clustering on the Observed Data points'
          return self.model.labels_
          
      def predict(self, Xdata):   
          cindex=self.model.fit_predict(Xdata)
          return cindex