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


# In[5]:


def learn_classifier(training_features, training_labels):
    classifier = RandomForestClassifier(n_estimators = 100)
#     classifier = svm.SVC(probability=True)
    classifier.fit(training_features, training_labels)
#     folds = range(5,31, 1)
#     #evalcrossvaluation
#     # evaluate each k value
#     for k in folds:
#     # define the test condition
#         cv = KFold(n_splits=k, shuffle=True, random_state=10)
#         # record mean and min/max of each set of results
#         k_mean, k_min, k_max = evaluate_model(cv,training_features,training_labels, classifier)
#         # report performance
#         print('-> folds=%d, accuracy=%.3f (%.3f,%.3f)' % (k, k_mean, k_min, k_max))
    return classifier


# In[6]:


# evaluate the model
def evaluate_model(cv, X, y, model):
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # return scores
    return mean(scores), scores.min(), scores.max()


# In[7]:


def compute_scores_training_set_sqsi(classifier, training_features, training_labels):
    countEvents = len(training_features)
    cv = KFold(n_splits=20, shuffle=True)

    scores_training_set= cross_val_predict(classifier, training_features, training_labels, cv=cv, n_jobs=-1, method='predict_proba')
    return scores_training_set


# In[8]:


# RUN DETECTOR


# In[9]:


def sqsi(training_data, testing_features, encoder, scaler):
    training_features = training_data[0]
    training_labels = training_data[1]
    size_training = len(training_features)
    all_features = training_features.append(testing_features)
    all_features_ready = prepare_features(all_features, training_labels, size_training, encoder, scaler)
    training_features = all_features_ready.iloc[0:size_training, :]
    testing_features = all_features_ready.iloc[size_training:,]
    
    classifier = learn_classifier(training_features, training_labels)
    scores_training_set = compute_scores_training_set_sqsi(classifier, training_features, training_labels)
    detected = False

    predict_probs_batch = classifier.predict_proba(testing_features)
    probs_batch = predict_probs_batch[:,1]
    probs_training = scores_training_set[:,1]
    p_value = stats.ks_2samp(probs_batch,probs_training)[1]
    if(p_value < 0.001):
        print('detected!')
        detected = True
        
    if (detected is False):
        print('no concept drift detected')
    
    return detected

