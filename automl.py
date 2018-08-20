from datetime import datetime

import pandas as pd
import numpy as np

import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_absolute_error,log_loss,auc
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import general_codes.run_regression as reg
import pdb
from sklearn import preprocessing
import csv
import time
import sys
from boostaroota import BoostARoota
import boost_test
from bayes_opt import BayesianOptimization
timestr = time.strftime("%Y%m%d-%H%M%S")
from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *
import glob
import re
import pdb
import pyproj    
import shapely
import shapely.ops as ops
from shapely.geometry.polygon import Polygon
from functools import partial
from scipy import stats
import collections
from collections import Counter
from itertools import groupby
from statsmodels import robust
#sys.stdout = open('../logs/test'+timestr+'.log', 'a')
#==============================================================================
#  Functions
#==============================================================================

# Standardize features


def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])
    
    # subtracy mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
    
    return df
    
def _nominal_columns_to_numeric(train_df, test_df):
    for f in train_df.columns:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values.astype('str')) + list(test_df[f].values.astype('str')))
            train_df[f] = lbl.transform(list(train_df[f].values.astype('str')))
            test_df[f] = lbl.transform(list(test_df[f].values.astype('str')))
        
    return train_df, test_df
    

def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))
        

    #df = standardize(df)
    #print("After standardization {}".format(df.shape))
        
    # create dummy variables for categoricals
    #df = pd.get_dummies(df)
    #print("After converting categoricals:\t{}".format(df.shape))
    

    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})
    
    df.fillna(0, inplace=True)
    
    return df
    

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def direction_from_ezimuth(data):
    dir_=[]
    for i in range(len(data)):
        if (data[[i]] > 90) and (data[[i]]<=180) :
           dir_.append('S')
        if (data[[i]] > 0) and (data[[i]]<= 90):
           dir_.append('E')
        if (data[[i]]>180) and (data[[i]]<=270) :
           dir_.append('W')
        if (data[[i]]>270) and (data[[i]]<=360) :
           dir_.append('N')
    return  dir_  


def calculte_area(data):
    new_col = list(zip(data.iloc[:,0], data.iloc[:,1]))
    
    geom = Polygon(new_col)
    geom_area = ops.transform(
    partial(
        pyproj.transform,
        pyproj.Proj(init='EPSG:4326'),
        pyproj.Proj(
            proj='aea',
            lat1=geom.bounds[1],
            lat2=geom.bounds[3])),
    geom)
    return geom_area.area

def pairwise(iterable):
    from itertools import tee, izip
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def calculate_distance(data):
    iter_dist_=[]

    for (i1, row1), (i2, row2) in pairwise(data.iterrows()):
        
        iter_dist_.append(haversine_np(row1[0],row1[1],row2[0],row2[1]))
        #print i1, i2, row1["value"], row2["value"]
    return sum(iter_dist_)
def sumPairs(arr, n):
 
    # final result
    sum = 0
    for i in range(n - 1, -1, -1):
        sum += i*arr[i] - (n-1-i) * arr[i]
    return sum
def mean_lat_lon(data):
    x=[]
    y=[]
    z=[]

    for idx,row in data.iterrows():
        
        x.append(np.sin(row[1]) * np.cos(row[0]))
        y.append(np.sin(row[1]) * np.sin(row[0]))
        z.append(np.cos(row[1])) 
    pdb.set_trace()
    x = float(np.sum(x))/np.shape(data)[0]
    y = float(np.sum(y))/np.shape(data)[0]
    z = float(np.sum(z))/np.shape(data)[0]


    lat_mean = np.arctan2(z, np.sqrt(x * x + y * y));
    lon_mean = np.arctan2(-y, x);
    return lat_mean,lon_mean

def calculate_features(od):
    data= pd.DataFrame()
    for k, v in od.items(): 
        #locals()[k] = v
        print(k)
        
        #Day_time
        #v = v[v.iloc[:,4]==1]
        #Night_Time
        #v = v[v.iloc[:,4]==0]

        min_lat = v.iloc[:,1:2].min().values[0]
        max_lat = v.iloc[:,1:2].max().values[0]
        min_lon = v.iloc[:,0:1].min().values[0]
        max_lon = v.iloc[:,0:1].max().values[0]
        #pdb.set_trace()
        avg_lat = v.iloc[:,1:2].mean().values[0]
        avg_lon = v.iloc[:,0:1].mean().values[0]

        avg_lat = v.iloc[:,1:2].mean().values[0]
        avg_lon = v.iloc[:,0:1].mean().values[0]

        #avg_lat,avg_lon = mean_lat_lon(v)

        #pdb.set_trace()

        var_lat = v.iloc[:,1:2].var().values[0]
        var_lon = v.iloc[:,0:1].var().values[0]
        

        all_groups=[list(group) for key,group in groupby(v.iloc[:,0:1].values.tolist())]
        binary_groups=[len(list(group))>1 for key,group in groupby(v.iloc[:,0:1].values.tolist())]
        duplicate_groups=[i for indx,i in enumerate(all_groups) if binary_groups[indx] == True]
        #pdb.set_trace()
        #sum(v.iloc[:,5:6][v.iloc[:,0:1].values==t[0]].values[0] for t in i) for indx,i in enumerate(duplicate_groups)]
        
        #stay_time = 
        #Neg_elevation_count = sum([key[0]<0 for key in v.iloc[:,3:4].values.tolist()])
        #Pos_elevation_count = sum([key[0]>0 for key in v.iloc[:,3:4].values.tolist()])
        #most_common,num_most_common = Counter(direction_from_ezimuth(v.iloc[:,2:3].values)).most_common(1)[0]
        #distance_for_speed=calculate_distance(v)
        #distance = 2 * haversine_np( v.iloc[:,0:1].min().values[0],v.iloc[:,1:2][v.iloc[:,0:1].values==v.iloc[:,0:1].min().values[0]].values[0][0],v.iloc[:,0:1].max().values[0], v.iloc[:,1:2][v.iloc[:,0:1].values==v.iloc[:,0:1].max().values[0]].values[0][0]) 
        distance = 2 * haversine_np( v.iloc[:,0:1].min().values[0],v.iloc[:,1:2].min().values[0],v.iloc[:,0:1].max().values[0], v.iloc[:,1:2].max().values[0]) 
        #speed = float(distance)/v.iloc[:,5:6].max().values[0]
        max_days =  v.iloc[:,7:8].max().values[0]
        Area_polygon =calculte_area(v)

        diff_in_lat = v.iloc[:,0:1].max().values[0] - v.iloc[:,0:1].min().values[0]
        diff_in_lon =  v.iloc[:,1:2].max().values[0] - v.iloc[:,1:2].min().values[0]

        v['new']=np.where(v.iloc[:,3]>0,'positive','Negative').tolist()
        positive_frame = v[v.iloc[:,3]>0]
        negative_frame = v[v.iloc[:,3]<0]
        #pdb.set_trace()
        if positive_frame.shape[0]!=0:
           pos_distance = 2 * haversine_np( positive_frame.iloc[:,0:1].min().values[0],positive_frame.iloc[:,1:2][positive_frame.iloc[:,0:1].values==positive_frame.iloc[:,0:1].min().values[0]].values[0][0],positive_frame.iloc[:,0:1].max().values[0], positive_frame.iloc[:,1:2][positive_frame.iloc[:,0:1].values==positive_frame.iloc[:,0:1].max().values[0]].values[0][0]) 
        else:
           pos_distance=0
        if negative_frame.shape[0]!=0: 
           neg_distance = 2 * haversine_np( negative_frame.iloc[:,0:1].min().values[0],negative_frame.iloc[:,1:2][negative_frame.iloc[:,0:1].values==negative_frame.iloc[:,0:1].min().values[0]].values[0][0],negative_frame.iloc[:,0:1].max().values[0], negative_frame.iloc[:,1:2][negative_frame.iloc[:,0:1].values==negative_frame.iloc[:,0:1].max().values[0]].values[0][0]) 
        else:
           neg_distance=0

        
        
        count_info=v.groupby([v['new'],(v['new']!=v['new'].shift()).cumsum()]).size().reset_index(level=1,drop=True)
        
        if max_days!=0:
            Area_polygon =calculte_area(v)/max_days
            avg_distance = float(distance)/max_days
            if pos_distance!=0:
               avg_pos_distance = float(pos_distance)/max_days
            else:
               avg_pos_distance=0

            if  avg_pos_distance!=0:
               avg_neg_distance = float(neg_distance)/max_days
            else:
               avg_neg_distance=0

            avg_ezimuth = float(v.iloc[:,2:3].mean().values[0])/max_days
            avg_elevation = float(v.iloc[:,3:4].mean().values[0])/max_days
            min_ezimuth = float(v.iloc[:,2:3].min().values[0])/max_days
            min_elevation = float(v.iloc[:,3:4].min().values[0])/max_days
            max_ezimuth = float(v.iloc[:,2:3].max().values[0])/max_days
            max_elevation = float(v.iloc[:,3:4].max().values[0])/max_days
            diff_in_azimuth = max_ezimuth - min_ezimuth
            diff_in_elevation = max_elevation - min_elevation
            #most_freq_daytime  =float(v.iloc[:,4:5].max().values[0])
            max_elapsed_time  = float(v.iloc[:,5:6].max().values[0])/max_days
            Neg_elevation_count = float(Counter(count_info.index)['Negative'])/max_days
            Pos_elevation_count = float(Counter(count_info.index)['Positive'])/max_days
            #speed = float(speed)/max_days
            #distance_for_speed = float(distance_for_speed)/max_days
            no_of_stays = float(sum([len(list(group))>1 for key, group in groupby(v.iloc[:,0:1].values.tolist())]))/max_days
            #pdb.set_trace()
        else:
            avg_distance = distance
            avg_pos_distance=pos_distance
            avg_neg_distance=neg_distance
            avg_ezimuth = v.iloc[:,2:3].mean().values[0]
            avg_elevation = v.iloc[:,3:4].mean().values[0]
            min_ezimuth = v.iloc[:,2:3].min().values[0]
            min_elevation = v.iloc[:,3:4].min().values[0]
            max_ezimuth = v.iloc[:,2:3].max().values[0]
            max_elevation = v.iloc[:,3:4].max().values[0]
            #most_freq_daytime  =v.iloc[:,4:5].max().values[0]
            max_elapsed_time  = float(v.iloc[:,5:6].max().values[0])
            Neg_elevation_count = Counter(count_info.index)['Negative']
            Pos_elevation_count = Counter(count_info.index)['Positive']
            diff_in_azimuth = max_ezimuth - min_ezimuth
            diff_in_elevation = max_elevation - min_elevation
            #speed = float(speed)
            no_of_stays = sum([len(list(group))>1 for key, group in groupby(v.iloc[:,0:1].values.tolist())])
            #distance_for_speed=float(distance_for_speed)
        v=[diff_in_azimuth,diff_in_elevation,diff_in_lat,diff_in_lon,min_lon,max_lon,min_lat,max_lat,avg_distance,max_ezimuth,max_elevation,min_ezimuth,min_elevation,avg_ezimuth,avg_elevation,max_elapsed_time,max_days,avg_lat,avg_lon,no_of_stays]



        #v=v.iloc[[0]]

        #if k =='original_ccp_combined':
        #pdb.set_trace()
        data = pd.concat([data,pd.DataFrame(v).T],axis=0)
    data.columns=['diff_in_azimuth','diff_in_elevation','diff_in_lat','diff_in_lon','min_lon','max_lon','min_lat','max_lat','avg_distance','max_ezimuth','max_elevation','min_ezimuth','min_elevation','avg_ezimuth','avg_elevation','max_elapsed_time','max_days','avg_lat','avg_lon','no_of_stays']
    return data
#==============================================================================
# Main
#==============================================================================
DATA_DIR = os.path.abspath('../data/train/')



#names=['longitude','latitude','azimuth','elevation','daytime','elapsedtime','clock','days']
#pdb.set_trace()
#Train Columns
dfs = { re.search('/([^/\.]*)\.csv', fn).group(1):
    pd.read_csv(fn,header=None,engine='python')for fn in glob.glob('../data/train/*.csv')}

od = collections.OrderedDict(sorted(dfs.items()))
train = calculate_features(od)


dfs = { re.search('/([^/\.]*)\.csv', fn).group(1):
    pd.read_csv(fn,header=None,engine='python',names=['longitude','latitude','azimuth','elevation','daytime','elapsedtime','clock','days'])for fn in glob.glob('../data/test/*.csv')}

#Test Columns
od = collections.OrderedDict(sorted(dfs.items()))
test = calculate_features(od)

train_label = pd.read_csv('../data/train_labels.csv',header=None)
train_label.columns=['Gender']

#pdb.set_trace()
train['Gender']= train_label.Gender.tolist()

train.to_csv('../data/train_first.csv',index=False)
test.to_csv('../data/test_first.csv',index=False)



paths = ['../data/train_first.csv','../data/test_first.csv']
target_name = "Gender"

rd = Reader(sep = ',')
df = rd.train_test_split(paths, target_name)


dft = Drift_thresholder()
df = dft.fit_transform(df)


opt = Optimiser(scoring = 'accuracy', n_folds = 5)

opt.evaluate(None, df)


space = {
       
        'ne__numerical_strategy':{"search":"choice","space" : [0, 'mean']},
        'ce__strategy':{"search":"choice",
                        "space":["label_encoding","random_projection", "entity_embedding","dummification"]}, 
        'fs__strategy' : {"space" : ["variance"]},
        'fs__threshold':{"search":"uniform",
                        "space":[0.01,0.3]}, 
           
        'est__max_depth':{"search":"choice",
                                 "space":[3,4,5,6,7]},
       'est__subsample' : {"search" : "uniform", "space" : [0.6,0.9]},
        'est__colsample_bytree' : {"search" : "uniform", "space" : [0.6,0.9]}
        }

best = opt.optimise(space, df,5)

prd = Predictor()
prd.fit_predict(best, df)
 
