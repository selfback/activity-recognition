#from __future__ import division

import selfback_utils as su
import scipy
import numpy as np
from detect_peaks import detect_peaks
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt


def read_data(path): 
    person_data = {}
    classes = os.listdir(path)
   
    for _class in classes:        
        files = os.listdir(path+_class)
        for f in files:
            p = f[:f.index('.')]           
            if f.endswith('.csv'):
                df = pd.read_csv(path+_class+'/'+f, header=0)                           
                df['class'] = _class              
                activity_data = {}
                if p in person_data:
                    activity_data = person_data[p] 
                    if _class in activity_data:
                        df = activity_data[_class].append(df)
                activity_data[_class] = df                
                person_data[p] = activity_data
    return person_data  
           
        
#Step counting algorithms based on frequency analysis
def count_steps1(rms, samp_rate):
    rms2 = su.filter(rms, samp_rate, 20)
    f = np.abs(np.real(scipy.fftpack.fft(rms2)))
    _f = f[1:]
    freq = np.fft.fftfreq(len(rms), d=(1.0/samp_rate))
    _freq = freq[1:]
    
    time = len(rms)/samp_rate
    
    f_sorted = np.sort(_f[0:len(_f)/2])
    top_two = f_sorted[-2:]
    #print top_two
    peak1 = top_two[1]
    peak2 = top_two[0]
    indx_p1 = np.where(_f == peak1)[0][0]
    indx_p2 = np.where(_f == peak2)[0][0]
    f1 = _freq[indx_p1]
    f2 = _freq[indx_p2]
    fh = f1    
    if f2 > f1:
        if peak2 >= (0.75 * peak1):
            if (1.5 * f1) < f2 and f2 < (2.5 * f1):
                fh = f2                
                
    steps1 = fh * time  
    return steps1
    

#Step counting algorithm based on peak counting    
def count_steps2(x, samp_rate):
    x1 = su.filter(x, samp_rate, 2)
    peaks = detect_peaks(x1, mph=1)
    return len(peaks)


if __name__  == '__main__':
    rgu_path = 'C:/Datasets/Activity/RGU/v2/Steps/'
    person_data = read_data(rgu_path)
    
    samp_rate = 100
    time_window = 8
    
    #windows = [5,6,7,8,9,10]
    windows = [8]
    for time_window in windows:
        true_steps = []
        pred_steps1 = []
        pred_steps2 = []
        for user in person_data:    
            user_steps = int(user[user.index('_')+1:])
            user_id = user[:user.index('_')]
            print user_id+': %d\n'%user_steps
            data = person_data[user]            
            df = data['walking']        
            df = su.smooth(df)
            #rms = su.rms_vector(df)
            rms = su.mag(df['x'], df['y'], df['z'])
            
            steps1 = count_steps1(rms, samp_rate)
            steps2 = count_steps2(rms, samp_rate)        
            print '\tSteps1: %d'%(steps1)
            print '\tSteps2: %d'%(steps2)
            
            true_steps.append(user_steps)
            pred_steps1.append(steps1)
            pred_steps2.append(steps2)
            
        print '============= '+str(time_window)+' ================================================='
        
        mse1 = mean_squared_error(true_steps, pred_steps1)
        mse2 = mean_squared_error(true_steps, pred_steps2)    
        
        print 'RMSE1: %f'%sqrt(mse1)
        print 'RMSE2: %f'%sqrt(mse2)
        print '\n'
        


