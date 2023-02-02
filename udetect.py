#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import sys
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import preprocessing
from scipy import stats
from scipy.stats import ks_2samp
import pandas as pd
from scipy.io.arff import loadarff 
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import time
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from numpy import mean
from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import math
from river import drift


# # Data Modeling

# In[2]:


def read_dataset(path, label):
    dataframe = pd.DataFrame()
    if 'synthetic' in path:
        data = loadarff(path)
        dataframe = pd.DataFrame(data[0])
    if 'real-world' in path:
        dataframe = pd.read_csv(path)
        
    labels = dataframe.loc[:, dataframe.columns == label]
    features = dataframe.loc[:, dataframe.columns != label]
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    return features, labels
    


# In[3]:


def prepare_features(features, training_labels, size_training, features_encoder, features_scaler): 
    if(features_encoder is None):
        features = features._get_numeric_data()
    else:
        cols = features.columns
        numerical_cols = features._get_numeric_data().columns
        categorical_cols = list(set(cols) - set(numerical_cols))
        
        if(isinstance(features_encoder, OneHotEncoder)):
            feature_arr = features_encoder.fit_transform(features[categorical_cols])
            feature_labels =  features_encoder.get_feature_names(categorical_cols)
            encoded_features = pd.DataFrame(feature_arr.toarray(), columns=feature_labels)
            
        if(isinstance(features_encoder, OrdinalEncoder)):
            feature_arr = features_encoder.fit_transform(features[categorical_cols])
            encoded_features = pd.DataFrame(feature_arr, columns=categorical_cols)
            
        if(isinstance(features_encoder, TargetEncoder)):
            transform = features_encoder.fit_transform(features[categorical_cols].iloc[:size_training], training_labels)
            training_encoded = pd.DataFrame(transform, columns=categorical_cols)
            testing_encoded = pd.DataFrame(features_encoder.transform(features[categorical_cols].iloc[size_training:len(features)]), columns=categorical_cols)
            encoded_features = training_encoded.append(testing_encoded)
            
        features = features._get_numeric_data().join(encoded_features)
         
    if(features_scaler is True):
        scaler = MinMaxScaler()
        features_training = scaler.fit_transform(features.iloc[:size_training])
        features_testing = scaler.transform(features.iloc[size_training:len(features)])
        features_training_df = pd.DataFrame(features_training, columns=features.columns)
        features_testing_df = pd.DataFrame(features_testing, columns=features.columns)
        features = features_training_df.append(features_testing_df)
        
    if(features_encoder is not None):
        print('categorical features encoder', features_encoder.__class__.__name__)
    else:
        print('no features encoder')
    if(features_scaler is True):
        print('features scaled using MinMaxScaler')
    else:
        print('features are not scaled')
        
    return features


# In[4]:


def get_training_data(features, labels, size_training): 
    training_features = features.iloc[:size_training]
    training_labels = labels[:size_training]
    return [training_features, training_labels]


# ## UDetect

# In[7]:


def window_summary(features):
    centroid = features.sum(axis=0) / len(features)
    sum_differences = 0
    features = features.values    
    for feature in features:
        sum_differences = sum_differences + np.linalg.norm(feature - centroid.values)
    mean_E_d = sum_differences / (len(features) - 1)
    
    return mean_E_d


# In[8]:


def change_parameter(summary_arr):
    training_mean = np.mean(summary_arr)
    std_dev = np.std(summary_arr)
    number_subgroups = len(summary_arr)
    R_sum = 0
    for i in range(number_subgroups - 1):
        R_sum = R_sum + np.linalg.norm(summary_arr[i + 1] - summary_arr[i])
    R_mean = R_sum / (number_subgroups - 1)
    A2 = 0.1
    #print(R_mean)
    #subgroup = 25
    LCL = training_mean -  A2 * R_mean
    UCL = training_mean +  A2 * R_mean
    print('lcl, ucl', LCL, UCL)
    return training_mean, UCL, LCL


# In[33]:


# RUN DETECTOR
def udetect(training_data, testing_features, encoder, scaler):
    training_features = training_data[0]
    training_labels = training_data[1]
    size_training = len(training_features)
    all_features = training_features.append(testing_features)
    all_features_ready = prepare_features(all_features, training_labels, size_training, encoder, scaler)
    training_features = all_features_ready.iloc[0:size_training, :]
    testing_features = all_features_ready.iloc[size_training:,]
    
    detected_batches = []
    detected = False

    training_summaries = []
    size_subgroup = 25
    for i in range(0, size_training, size_subgroup):
        training_summaries.append(window_summary(training_features.iloc[i:i + size_subgroup]))
    training_mean, UCL_Ed, LCL_Ed = change_parameter(training_summaries)    

    testing_summaries = []
    for i in range(0, len(testing_features), size_subgroup):
        testing_summaries.append(window_summary(testing_features.iloc[i:i + size_subgroup]))
    summary_batch = np.mean(testing_summaries)

    if(summary_batch < LCL_Ed or summary_batch > UCL_Ed):
            print('detected!')
            detected = True

    if (detected is False):
        print('no concept drift detected')
    
    return detected

