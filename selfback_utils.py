from __future__ import division
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import butter, lfilter
from scipy import fftpack
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from dtw import dtw
import math

feats = ['sum_x', 'sum_y', 'sum_z', 'sum_m', 'mean_x', 'mean_y', 'mean_z', 'mean_m', 'sd_x', 'sd_y', 'sd_z', 'sd_m', 'iqr_x', 'iqr_y', 'iqr_z', 'iqr_m', 'p10_x', 'p25_x', 'p50_x', 'p75_x', 'p90_x', 'p10_y', 'p25_y', 'p50_y', 'p75_y', 'p90_y', 'p10_z', 'p25_z', 'p50_z', 'p75_z', 'p90_z', 'p10_m', 'p25_m', 'p50_m', 'p75_m', 'p90_m', 'p2p_x', 'p2p_y', 'p2p_z', 'p2p_m', 'pw_x', 'pw_y', 'pw_z', 'pw_m', 'lpw_x', 'lpw_y', 'lpw_z', 'lpw_m', 'acorr_x', 'acorr_y', 'acorr_z', 'acorr_m', 'kurt_x', 'kurt_y', 'kurt_z', 'kurt_m', 'skw_x', 'skw_y', 'skw_z', 'skw_m', 'corr_xy', 'corr_xz', 'corr_yz', 'mcross_x', 'mcross_y', 'mcross_z', 'mcross_m', 'energy_xyz', 'rsm_x', 'rsm_y', 'rsm_z', 'rsm_m', 'max_x', 'max_y', 'max_z', 'max_m', 'min_x', 'min_y', 'min_z', 'min_m', 'mad_x', 'mad_y', 'mad_z', 'mad_m']
features = pd.Series(feats)
_label_map = {'standing': 'sedentary', 'sitting':'sedentary', 'lying':'sedentary', 'jogging':'running', 'walk_slow':'walking', 'walk_mod':'walking', 'walk_fast':'walking', 'upstairs':'stairs', 'downstairs': 'stairs'} 

def find_claps(x):
    #print len(x)
    claps = []
    i=1
    start = x[0]   
    prev = start 
    count = 0
    while i < len(x):
        end = x[i]
        #print str(prev)+' '+str(end) 
        diff = end - prev       
        if diff > 15 and diff <= 100:
            count += 1                
        elif diff > 100:
            start = end  
        prev = end       
        if count == 2:
            ind = (start,end)
            claps.append(ind)
            count = 0
        i += 1
    #if prev != start:
        #ind = (start, end)
        #claps.append(ind)
    return claps
    
def find_segments(x, claps):
    i=0
    segments = []
    for c in claps:
        j = c[0]-200
        segments.append(x.iloc[i:j])
        i = c[1]+300
    segments.append(x.iloc[i:len(x)])
    return segments
    
def read_train_data(path):     
    #colnames = ['time', 'x', 'y', 'z']
    person_data = {}
    classes = os.listdir(path)
   
    for _class in classes:        
        files = os.listdir(path+_class)
        for f in files:
            p = f[:f.index('_')]
            #print p
            if f.endswith('.csv'):
                df = pd.read_csv(path+_class+'/'+f, header=0)
                #print df.head()                  
                df['class'] = _class                    
                #print p
                #print '\t' + _class + ' ' + str(len(df))
                #print '\n'
                activity_data = {}
                if p in person_data:
                    activity_data = person_data[p] 
                    if _class in activity_data:
                        df = activity_data[_class].append(df)
                activity_data[_class] = df
                #print '\t'+ str(activity_data)
                person_data[p] = activity_data
    return person_data  
    
    
def filter(x, samp_rate, high, order=5):
      nyq = 0.5 * samp_rate    
      high = high/nyq
      b, a = butter(order, high, btype='low')
      y = lfilter(b, a, x)
      return y
     
def sum(x):
    return np.sum(x)

def mean(x):
    return np.mean(x)

def sd(x):
    return np.std(x)

def iqr(x):
    return np.subtract(*np.percentile(x,[75,25]))
    
def percentile(x, p):
    return np.percentile(x, p)

def peak2peak_amp(x):
    return np.max(x) - np.min(x)

def power(x):
    return np.sum(np.square(x))
    
def log_power(x):
    printed = False
    mf = 0.00001
    x = np.add(x,mf)
    log_power = np.sum(np.log(np.square(x)))
    if not printed:
        if 0 in x:
            print x
            printed = True
    return log_power

def lag_one_autocorr(x, m=None):
    num = 0
    if m is None:
        m = np.mean(x)
    i = 0    
    while i < len(x)-1:
        num += (x[i] - m) * (x[i+1] - m)
        i += 1
    denom = np.sum(np.square(np.subtract(x,m)))
    return num/denom

def kurtosis(x):
    return stats.kurtosis(x)

def skewness(x, m=None):
    if m is None:
        m = np.mean(x)
    num = np.sum(np.power(np.subtract(x,m),3))/len(x)
    denom = np.power(np.sqrt(np.sum(np.square(np.subtract(x,m)))/(len(x)-1)), 3)
    return num/denom

def corr(a,b):  
    return np.corrcoef(a,b)[0,1]

def zero_cross(x):
    count = 0
    median = np.median(x)
    last_sign = np.sign(x[0])
    i = 1
    while i < len(x):
        if np.sign(x[i]) != last_sign:
            last_sign = np.sign(x[i])
            count +=1
        i+=1
    return count

def median_cross(x, m=None):
    if m is None:
        m = np.median(x)
    count = 0
    i=0
    while i < len(x)-1:
        count += np.absolute(np.sign(x[i] - m) - np.sign(x[i+1] - m))
        i+=1
    return count/2
        
#New features
def mag(x, y, z):
    x_2 = np.square(x)
    y_2 = np.square(y)
    z_2 = np.square(z)
    mag =  np.sqrt(np.add(x_2, np.add(y_2,z_2)))   
    return mag
    
def rho(x, y, z):
    theta = (180/math.pi) * math.atan(x/(math.sqrt(math.pow(y, 2) + math.pow(z,2))))
    return theta

def phi(x, y, z):
    theta = (180/math.pi) * math.atan(y/(math.sqrt(math.pow(x, 2) + math.pow(z,2))))
    return theta
    
def theta(x, y, z):
    theta = (180/math.pi) * math.atan(z/(math.sqrt(math.pow(y, 2) + math.pow(x,2))))
    return theta
    
def energy(x, y, z):
    ex = np.sqrt(np.sum(np.square(np.subtract(x,mean(x)))))
    ey = np.sqrt(np.sum(np.square(np.subtract(y,mean(y)))))
    ez = np.sqrt(np.sum(np.square(np.subtract(z,mean(z)))))
    
    e = (1/(3 * len(x))) * (ex + ey + ez)
    return e
    
def root_square_mean(x):
    return np.sqrt(np.mean(np.square(x)))
    
def mean_abs_dev(x):
    m = np.mean(x)
    return np.mean(np.subtract(x, m))
    
def spec_centroid(x):
    i = range(x)
    return np.sum(np.multiply(x, i))/np.sum(x)
    
def spec_entropy(x):
    N = len(x)
    p = np.divide(np.square(x), N)
    pi = np.divide(p, np.sum(p))
    H = - np.sum(np.multiply(pi, np.log(pi)))
    if math.isnan(H):
        print p
        print pi
        print H
    return H
    
    
time_feats = [sum, mean, sd, iqr, percentile, peak2peak_amp, power, log_power, lag_one_autocorr, kurtosis, skewness, corr, zero_cross, energy, root_square_mean, mean_abs_dev]
     

# In[5]:

def read_csv_data(path, colnames=None, header=0):
    df = pd.read_csv(path, header=header, names = colnames)
    return df


# In[7]:
def rms_vector(tw):
    x = tw['x'].values
    y = tw['y'].values
    z = tw['z'].values
    
    rms = []
    for i in range(len(x)):  
        val = (x[i] * x[i]) + (y[i] * y[i]) +(z[i] * z[i])
        val = np.sqrt(val)/3
        rms.append(val)
    rms = np.array(rms)
    return rms


def get_freq_features(tw):
    features = []
    x = tw['x'].values
    y = tw['y'].values
    z = tw['z'].values 
    m = mag(x,y,z)  
    mf = 0.00001
    
    fftx = np.add(np.abs(np.real(np.fft.fft(x))), mf)
    ffty = np.add(np.abs(np.real(np.fft.fft(y))), mf)
    fftz = np.add(np.abs(np.real(np.fft.fft(z))), mf)
    
    fftx = fftx[0:int(len(fftx)/2)]
    ffty = ffty[0:int(len(ffty)/2)]
    fftz = fftz[0:int(len(fftz)/2)]    
    
    #print fftx
    #print '\n'
    #print ffty
    #print '\n'
    #print fftz
    
    #features.append(sum(fftx))
    #features.append(sum(ffty))
    #features.append(sum(fftz))
    
    features.append(mean(fftx))
    features.append(mean(ffty))
    features.append(mean(fftz))
    
    features.append(sd(fftx))
    features.append(sd(ffty))
    features.append(sd(fftz))
    
    features.append(np.max(fftx))
    features.append(np.max(ffty))
    features.append(np.max(fftz))
    
    features.append(np.median(fftx))
    features.append(np.median(ffty))
    features.append(np.median(fftz))    
    
    features.append(spec_entropy(fftx))
    features.append(spec_entropy(ffty))
    features.append(spec_entropy(fftz))
    
    return features
    
def get_time_features(tw, indx):
    features = []
    x = tw['x'].values
    y = tw['y'].values
    z = tw['z'].values 
    m = mag(x,y,z) 
        
    ax = x[int(len(x)/2)]
    ay = y[int(len(y)/2)]
    az = z[int(len(z)/2)]    
    
    if indx == 0:
        features.append(sum(x)) #1
        features.append(sum(y)) #2
        features.append(sum(z)) #3
        features.append(sum(m))    
    elif indx == 1:
        features.append(mean(x)) #4
        features.append(mean(y)) #5
        features.append(mean(z)) #6
        features.append(mean(m))     
    elif indx == 2:
        features.append(sd(x)) #7
        features.append(sd(y)) #8 H
        features.append(sd(z)) #9
        features.append(sd(m)) 
    elif indx == 3:
        features.append(iqr(x)) #10
        features.append(iqr(y)) #11 H
        features.append(iqr(z)) #12
        features.append(iqr(m)) 
    elif indx == 4:
        features.extend(percentile(x, [10,25,50,75,90])) #13 - 17
        features.extend(percentile(y, [10,25,50,75,90])) #18 - 22
        features.extend(percentile(z, [10,25,50,75,90])) #23 - 27
        features.extend(percentile(m, [10,25,50,75,90]))
    elif indx == 5:
        features.append(peak2peak_amp(x)) #28
        features.append(peak2peak_amp(y)) #29 
        features.append(peak2peak_amp(z)) #30
        features.append(peak2peak_amp(m)) 
    elif indx == 6:
        features.append(power(x)) #31
        features.append(power(y)) #32 H
        features.append(power(z)) #33
        features.append(power(m))
    elif indx == 7:
        features.append(log_power(x)) #34
        features.append(log_power(y)) #35 H
        features.append(log_power(z)) #36
        features.append(log_power(m))
    elif indx == 8:
        features.append(lag_one_autocorr(x)) #37
        features.append(lag_one_autocorr(y)) #38
        features.append(lag_one_autocorr(z)) #39 H
        features.append(lag_one_autocorr(m))
    elif indx == 9:
        features.append(kurtosis(x)) #40
        features.append(kurtosis(y)) #41
        features.append(kurtosis(z)) #42
        features.append(kurtosis(m))
    elif indx == 10:
        features.append(skewness(x)) #43
        features.append(skewness(y)) #44
        features.append(skewness(z)) #45
        features.append(skewness(m))
    elif indx == 11:
        features.append(corr(x,y)) #46
        features.append(corr(x,z)) #47
        features.append(corr(y,z)) #48 H
    elif indx == 12:
        features.append(zero_cross(x)) #49
        features.append(zero_cross(y)) #50 H
        features.append(zero_cross(z)) #51
        features.append(zero_cross(m))
    elif indx == 13:
        features.append(energy(x,y, z))
    elif indx == 14:
        features.append(root_square_mean(x))
        features.append(root_square_mean(y))
        features.append(root_square_mean(z))
        features.append(root_square_mean(m))
    elif indx == 15:
        features.append(np.max(x))
        features.append(np.max(y))
        features.append(np.max(z))
        features.append(np.max(m))
    elif indx == 16:
        features.append(np.min(x))
        features.append(np.min(y))
        features.append(np.min(z))        
        features.append(np.min(m))
    elif indx == 17:
        features.append(mean_abs_dev(x))
        features.append(mean_abs_dev(y))
        features.append(mean_abs_dev(z))
        features.append(mean_abs_dev(m))
    elif indx == 18:
        features.append(rho(ax, ay, az))
        features.append(phi(ax, ay, az))
        features.append(theta(ax, ay, az))
    
    return features

def extract_features(time_windows):   
    data = []
    _class = []
    #print len(time_windows)   
    for tw in time_windows:
        features = []
        
        #features.extend(get_freq_features(tw))
        
        features.extend(get_time_features(tw))        
        
        data.append(features)     
        
        _class.append(tw['class'].iloc[0])

    data_set = np.array(data)
 
    return data_set, _class
    
def extract_features2(time_windows, feautures):
    data = []
    _class = []
    #print len(time_windows)   
    for tw in time_windows:
        feat_vals = []
        
        feat_vals.extend(get_freq_features(tw, features))
        
        feat_vals.extend(get_time_features(tw, features))        
        
        data.append(feat_vals)     
        
        _class.append(tw['class'].iloc[0])
    
    data_set = np.array(data)    
    return data_set, _class


def extract_dct_features(time_windows, class_attr=None, n_comps=48):
    X_matrix = []
    y_vect = None
    if class_attr is not None:
        y_vect = []

    for tw in time_windows:        
        x = tw['x'].values
        y = tw['y'].values
        z = tw['z'].values
        m = mag(x,y,z)
        
        dct_x = np.abs(fftpack.dct(x))
        dct_y = np.abs(fftpack.dct(y))
        dct_z = np.abs(fftpack.dct(z))
        dct_m = np.abs(fftpack.dct(m))
        
        v = np.array([])       
        v = np.concatenate((v, dct_x[:n_comps]))            
        v = np.concatenate((v, dct_y[:n_comps]))
        v = np.concatenate((v, dct_z[:n_comps]))
        v = np.concatenate((v, dct_m[:n_comps]))       
        X_matrix.append(v)
        if y_vect is not None:
            y_vect.append(tw[class_attr].iloc[0])       
    
    X_matrix = np.array(X_matrix) 
           
    if y_vect is None:
        return X_matrix
    else:
        return X_matrix, y_vect
    
def extract_dst_features(time_windows, class_attr, n_comps=90):
    X_matrix = []
    y_vect = None
    if class_attr is not None:
        y_vect = []
    #n_comps = 50
    for tw in time_windows:        
        x = tw['x'].values
        y = tw['y'].values
        z = tw['z'].values
        m = mag(x,y,z)
        
        dst_x = np.abs(fftpack.dst(x))
        dst_y = np.abs(fftpack.dst(y))
        dst_z = np.abs(fftpack.dst(z))
        dst_m = np.abs(fftpack.dst(m))
        
        v = np.array([])       
        v = np.concatenate((v, dst_x[:n_comps]))            
        v = np.concatenate((v, dst_y[:n_comps]))
        v = np.concatenate((v, dst_z[:n_comps]))
        v = np.concatenate((v, dst_m[:n_comps]))
        X_matrix.append(v)
    X_matrix = np.array(X_matrix)      
    
    if y_vect is not None:
        y_vect.append(tw[class_attr].iloc[0])     
    
    return X_matrix, y_vect
    
    
def extract_fft_features(time_windows, n_comps=48):
    X_train = []
    y_train = []
    #n_comps = 50
    for w in time_windows:
        x = w['x'].values
        y = w['y'].values
        z = w['z'].values
        m = mag(x,y,z)
        
        fftx = np.abs(np.real(np.fft.fft(x)))
        ffty = np.abs(np.real(np.fft.fft(y)))
        fftz = np.abs(np.real(np.fft.fft(z)))  
        fftm = np.abs(np.real(np.fft.fft(m)))       
  
        v = fftx[:n_comps]               
        v = np.concatenate((v, ffty[:n_comps]))       
        v = np.concatenate((v, fftz[:n_comps]))            
        v = np.concatenate((v, fftm[:n_comps]))
        
        X_train.append(v)
        y_train.append(w['class'].iloc[0])
    X_train = np.array(X_train)
    return X_train, y_train
    
    
def extract_fft_dct_features(time_windows, n_comps=48):
    X_train = []
    y_train = []
    #n_comps = 50
    for w in time_windows:
        x = w['x'].values
        y = w['y'].values
        z = w['z'].values
        m = mag(x,y,z)
        
        fftx = np.abs(np.real(np.fft.fft(x)))
        ffty = np.abs(np.real(np.fft.fft(y)))
        fftz = np.abs(np.real(np.fft.fft(z)))  
        fftm = np.abs(np.real(np.fft.fft(m))) 
        
        dct_x = np.abs(fftpack.dct(x, norm='ortho'))
        dct_y = np.abs(fftpack.dct(y, norm='ortho'))
        dct_z = np.abs(fftpack.dct(z, norm='ortho'))
        dct_m = np.abs(fftpack.dct(m, norm='ortho'))
        
        v = dct_x[:n_comps]              
        v = np.concatenate((v, dct_y[:n_comps]))               
        v = np.concatenate((v, dct_z[:n_comps])) 
        v = np.concatenate((v, dct_m[:n_comps]))
        
             
  
        #v = fftx[:n_comps]  
        v = np.concatenate((v, fftx[:40]))             
        v = np.concatenate((v, ffty[:40]))       
        v = np.concatenate((v, fftz[:40]))            
        v = np.concatenate((v, fftm[:40]))
        
        X_train.append(v)
        y_train.append(w['class'].iloc[0])
    X_train = np.array(X_train)
    return X_train, y_train
  
      
def extract_dct_mag_features(time_windows, n_comps=48):
    X_train = []
    y_train = []
    #n_comps = 50
    for w in time_windows:
        x = w['x'].values
        y = w['y'].values
        z = w['z'].values
        
        m = mag(x,y,z)
        
        dct = fftpack.dct(m, norm='ortho')
        dct = np.abs(dct)
        v = dct[:n_comps]    
        
        X_train.append(v)
        y_train.append(w['class'].iloc[0])
    X_train = np.array(X_train)
    return X_train, y_train


# In[31]:

def split_by_class2(df):
    class_labels = df['class']
    class_df = {}
    unique_labels = class_labels.unique()    
    for l in unique_labels:
        class_df[l] = df[df['class']==l]
    return class_df

def split_windows(_df, samp_rate, w, overlap_ratio=None, overlap_time=None, min_width=1):
    time_windows = []
    width = samp_rate * w  
    increment = width
    if overlap_time is not None:
        increment = samp_rate * overlap_time
    elif overlap_ratio is not None:
        increment = int(width * (1-overlap_ratio))
        
    i = 0
    N = len(_df.index)
    while i < N:  
        start = i
        end = start+width
        if end > N:
            end = N
        elif (N - end) < samp_rate * min_width:
            end = N 
        #print str(start)+" "+str(end)
        tw = _df.iloc[start:end]
        classlabels = tw['class'].unique()
        if len(classlabels) == 1:
            time_windows.append(tw)
    
        increment = end-start
        i = int(i + (increment))
    return time_windows

    
# Resample signal to lower sample rate    
def resample(df, orig_rate, new_rate):
    time_windows = split_windows(df, orig_rate, 1)
    tw = time_windows[0]
    #print tw
    ind = [i * int(len(tw)/new_rate) for i in range(0,new_rate)]
    #print i
    new_df = tw.iloc[ind]
    for i in range(1,len(time_windows)):
        tw = time_windows[i]
        ind = [i * int(len(tw)/new_rate) for i in range(0,new_rate)]
        new_df.append(tw.iloc[ind])
    #print new_df
    return new_df
    
def maj_vote(pred_labels):
    preds = pd.Series(pred_labels)
    groups = preds.groupby(preds).count()
    class_dict = groups.to_dict()
    #print '\t'+str(class_dict)
    pred_class = max(class_dict, key=class_dict.get)
    #print '\t('+pred_class+')'
    return pred_class
    
def relieff(X,y, ind, _class, no_iter, nneighbors):
    df = pd.DataFrame(X)
    df['class'] = y
    knn = KNeighborsClassifier(n_neighbors=nneighbors)
    #group dataframe by class
    grouped = df.groupby(_class)
    
    #extract dataframe groups
    by_class_dfs = [df for name, df in grouped]
    
    #Retrieve selected class dataframe   
    df0 = by_class_dfs.pop(ind)
    pos_class = df0['class'].iloc[0]   
    
    #convert dataframe for each group into X matrices of values and y vectors of class labels
    Xy_list = [df_to_Xy(df) for df in by_class_dfs]
    
    #train Knn models
    knn_models = []
    for X, y in Xy_list:
        knn = KNeighborsClassifier()
        knn_models.append(knn.fit(X, y))
   
    num_attrs = len(df0.sample().columns)-1
    weights = np.zeros(num_attrs)
    
    num_classes = len(by_class_dfs)
    i = 0
    while i < no_iter:
        #print 'i: '+str(i)
        inst0 = df0.drop('class', axis=1).sample()
        a = inst0.as_matrix()[0]
        _df0 = df0.drop(inst0.index)
        X0, y0 = df_to_Xy(_df0)
        knn0 = KNeighborsClassifier(n_neighbors=nneighbors)
        knn0 = knn0.fit(X0, y0)
        nn_hit_indices = knn0.kneighbors(a.reshape([1,-1]), return_distance=False)[0]
        
        nn_misses = [knn.kneighbors(a.reshape([1,-1]), return_distance=False)[0] for knn in knn_models]       
                    
        for j in range(num_attrs):   
            nn_hit = [abs(a[j] - X0[l][j]) for l in nn_hit_indices]
            nn_hit_val = np.mean(nn_hit)    
              
            nn_miss = np.zeros(num_classes)            
            #print '\tj: '+str(j)
            #For each negative class
            for k in range(num_classes):
                #printed = False
                #print '\t\tk: '+str(k)
                X1 = Xy_list[k][0]
                #For each neighbor
                nn_miss_k = [abs(a[j] - X1[l][j]) for l in nn_misses[k]]                
                #if printed == False:
                    #print '\t\t\tl: '+str(l)
                    #printed = True
                nn_miss[k] = np.mean(nn_miss_k)  
            nn_miss_val = np.mean(nn_miss)    
            weights[j] = weights[j] - nn_hit_val + nn_miss_val
        i+=1            
    return weights, pos_class

            
def df_to_Xy(df):
    labels = df['class']
    y = labels.values
    df = df.drop('class', axis=1)
    X = df.as_matrix() 
    return X, y

def relabel(_class, label_map=_label_map):    
    _class_new = [label_map[label] if label in label_map.keys() else label for label in _class]    
    return _class_new    
    
def change_class_labels(df, label_map):
    _class = relabel(df['class'], label_map) 
    _df = df.drop('class', axis=1)
    _df['class'] = _class
    return _df
    
def select_features(X, weights, n):
    _weights = pd.Series(weights)
    df = pd.DataFrame(X)
    
    _weights_sorted = _weights.sort_values(ascending=False)
    indices = _weights_sorted.keys()
    sel_indices = indices[0:n]
    sel_df = df.iloc[:, sel_indices]
    return sel_df.as_matrix()
    
def dtw_sim(a,b):
    dist, cost, path = dtw(a,b)
    return 1/dist
    
def f1_score(y_true, y_pred):
    _f1 = 0.0
    labels_df = pd.DataFrame({'labels':y_true, 'counts':np.ones(len(y_true))})                
    label_counts = labels_df.groupby('labels').count().to_dict()['counts']    
    labels = label_counts.keys()    
    if len(labels) > 2:
        _f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    else:
        f1_0 = metrics.f1_score(y_true, y_pred, pos_label=labels[0], average='binary')
        f1_1 = metrics.f1_score(y_true, y_pred, pos_label=labels[1], average='binary')
        w0 = label_counts[labels[0]]
        w1 = label_counts[labels[1]]
        _f1 = ((w0 * f1_0) + (w1 * f1_1))/(w0 + w1)
    return _f1

def wrapper_select_features(X_train_df, y_train, X_val_df=None, y_val=None):
    sel_feats = []
    f1 = 0.0
    _features = X_train_df.columns
    #print 'Features: '+str(features)
    _features = np.random.choice(_features, len(_features), replace=False)
    #print '\nFeatures: '+str(features)
    
    if X_val_df is None:
        X_val_df = X_train_df
        y_val = y_train
    
    sel_feats.extend(_features)   
    
    clf = SVC()
    clf = clf.fit(X_train_df, y_train)
    preds = clf.predict(X_val_df)
    #preds = clf.predict(df)
    f1 = f1_score(y_val, preds)
    
    for i in range(len(_features)-1):                       
        f = [sel_feats.pop()]
        X_train_sel = X_train_df.iloc[:, sel_feats]
        X_val_sel = X_val_df.iloc[:, sel_feats]
        
        clf = SVC()
        clf = clf.fit(X_train_sel, y_train)
        
        preds = clf.predict(X_val_sel)
        #preds = clf.predict(X_train_sel)                
        
        _f1 = f1_score(y_train, preds)                
       
        if f1 > _f1:  
            #sel_feats.pop()                      
            f.extend(sel_feats)
            sel_feats = f
        else:   
            f1 = _f1                
        
    return sel_feats    

def smooth(tw, tau = 0.1):
    _tw = tw.copy()
    channels = ['x', 'y', 'z']    
    
    for ch in channels:
        s = _tw[ch]
        #print type(s)
        #print len(s)
        #print s.head()
        t = smooth_values(s.values, tau)
        _tw[ch] = t
    return _tw
    
def smooth_values(s, tau = 0.1):
    b = s[0]
    t = np.zeros_like(s)
    t[0] = b
    for i in range(1,len(s)):
        #print i
        if abs(s[i] - b) < tau:
            t[i] = b
        else:
            t[i] = s[i]
            b = s[i]
    return t
    