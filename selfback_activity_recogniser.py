
# coding: utf-8

# In[3]:
from __future__ import division
import os
import numpy as np
import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy import fftpack
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn import preprocessing as prep
import pickle
from sklearn.externals import joblib
from datetime import datetime

import selfback_utils as su
import selfback_stepcounter as sc


###################################################
########### ACTVITY RECOGNITION ##################
##################################################

class ActivityRecogniser:
    
    amb_activities = ['walking', 'stairs', 'running']
    classifier = None
    scaler = None

    def __init__(self, modelpath=None):
        if modelpath is not None:
            self.load_models(modelpath)
    
    def load_models(self, path):        
        self.classifier = joblib.load(path+'/classifier.pkl')
        self.scaler = joblib.load(path+'/scaler.pkl')
    
    
    def train_model(self, train_data, samp_rate=100, window_length=10):
        ########  Split train data into time windows  ############################################    
        time_windows_train = []
        for data in train_data:  
            #print len(data)
            for activity in data:
                df = data[activity]                    
                windows = su.split_windows(df, samp_rate, window_length, overlap_ratio=0.5)                    
                time_windows_train.extend(windows)    
        
        #########  Extract Features  #############################################            
        X_train, y_train = su.extract_dct_features(time_windows_train, class_attr='class')                 
        #X_train, y_train = su.extract_features(time_windows_train)    
        #n_features = X_train.shape[1]
        
        #############  Scale Features  ######################################
        self.scaler = prep.StandardScaler()
        #scaler = prep.MinMaxScaler()
        X_train_norm = self.scaler.fit_transform(X_train)
        
        #############  Apply PCA  ######################################
        #pca = PCA(n_components = int(n_features*n_comps))
        #X_train_norm = pca.fit_transform(X_train_norm)         
        
        ##########  Change Data Labels Granularity ##########################  
        y_train = su.relabel(y_train, label_map) 
                    
        #########  Train Classifier  #########################################
        clf = SVC()            
        clf = clf.fit(X_train_norm, y_train) 
        self.classifier = clf  
        
        ########  Persist Models  ########################################## 
        path = 'models'
        if not os.path.isdir(path):
            os.makedirs(path)
        joblib.dump(clf, path+'/classifier.pkl')
        joblib.dump(self.scaler, path+'/scaler.pkl')      
    
    
    def predict_activities(self, data, samp_rate=100, window_length=10, format_result=True):
        if self.classifier is not None:
            windows = su.split_windows(data, samp_rate, window_length, overlap_ratio=0.0)                            
            X_test = su.extract_dct_features(windows)           
            
            ##########  Scale Features #####################################
            X_test_norm = self.scaler.transform(X_test) 
            
            ##########  Apply PCA  ######################################### 
            #X_test_norm = pca.transform(X_test_norm)    
                                        
            pred_activities = self.classifier.predict(X_test_norm) 
            
            if not format_result:          
                return pred_activities
            else:
                return self.format_output(windows, pred_activities)
        else:
            raise InitError()
            
    
    def get_stepcounts(self, df, samp_rate=100):
        x = df['x']
        y = df['y']
        z = df['z']
        mag = su.mag(x,y,z)
        steps = sc.count_steps2(mag, samp_rate)
        return steps
            
    
    def format_output(self, windows, predictions):
        output = []
        df = windows[0]
        start_time = df['time'].iloc[0]
        end_time = df['time'].iloc[len(df)-1]
        activity = activity = predictions[0]
        temp = df
        for i in range(1, len(windows)): 
            df = windows[i] 
            if predictions[i] != activity:                
                if activity in self.amb_activities:
                    step_count = self.get_stepcounts(temp)              
                    output.append({'timestamp_start':start_time, 'timestamp_end':end_time, 'activity':activity, 'steps':step_count})
                else:
                    output.append({'timestamp_start':start_time, 'timestamp_end':end_time, 'activity':activity})
                temp = df
                activity = predictions[i]
                start_time = df['time'].iloc[0]
                end_time = df['time'].iloc[len(df)-1] 
            else:
                temp = pd.concat([temp, df])
                end_time = df['time'].iloc[len(df)-1]                         
        if activity in self.amb_activities:
            step_count = self.get_stepcounts(temp)              
            output.append({'timestamp_start':start_time, 'timestamp_end':end_time, 'activity':activity, 'steps':step_count})
        else:
            output.append({'timestamp_start':start_time, 'timestamp_end':end_time, 'activity':activity})
        return output
        
class InitError(Exception):    
    def __str__(self):
        return "Model not initialised. Please provide path to model in constructor or train new model using 'train_model' method."
    
    
if __name__ == '__main__':
    
    data_path = 'C:/Datasets/Activity/RGU/v2/Labelled_20/'
    person_data = su.read_train_data(data_path)
    
    sendentary = ['standing', 'sitting', 'lying']
    label_map = {'standing': 'standing', 'sitting':'sedentary', 'lying':'sedentary', 'jogging':'running', 'walk_slow':'walking', 'walk_mod':'walking', 'walk_fast':'walking', 'upstairs':'stairs', 'downstairs': 'stairs'} 
     
    
    exp_name = 'SVM_Flat_4class_Tw10_dct+mag_20data.txt'
    
    outfile = open('C:/results/'+exp_name, 'w')    
   
    acc_scores = []
    f1_scores = []
    tw_true_classes = []
    tw_pred_classes = []
    
    instance_ids = person_data.keys()
    N = len(instance_ids)
    instance_ids.extend(instance_ids)  
    
    ar = ActivityRecogniser('models')
    #ar = ActivityRecogniser()
    for ind in range(1):
        test_case = instance_ids[ind]        
        print 'Test case: '+test_case
        #test_case = 'person101'
        train_data = [value for key, value in person_data.items() if key not in [test_case]]
        test_data = person_data[test_case]        
        
        #############  Train Model  ##########################################        
        #ar.train_model(train_data)        
        
        ############  Classify Test Set  ####################################
        for activity in test_data:  
            print '#################    '+activity+'    ###################'
            df = test_data[activity]      
            predictions = ar.predict_activities(df)
            print predictions
