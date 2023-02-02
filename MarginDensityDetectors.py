#for performance:
# from sklearnex import patch_sklearn, config_context
# patch_sklearn()

# import warnings
# warnings.filterwarnings("error")

#approxximating
from sklearn.kernel_approximation import Nystroem

from sklearn import preprocessing
import pandas as pd
import sklearn.svm as svm
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from scipy.io import arff
import scipy as sp
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import *

import time
from sklearn.preprocessing import MinMaxScaler
from abc import ABC, abstractmethod
from sklearn import tree
from math import trunc
from math import ceil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score





##These are the Drift Detectors


class MDDriftDetector(ABC):

    def KXVal(self, data, labels,K,*classifierSettings):
        splitData=np.array_split(data,K)
        splitLabels=np.array_split(labels,K)
        accuracies = np.zeros(K)
        densities = np.zeros(K)
        for i in range(K):
            trainData = np.concatenate(splitData[:i]+splitData[i+1:])
            trainLabels= np.concatenate(splitLabels[:i]+splitLabels[i+1:])
            dens,acc= self.CrossValid(trainData,splitData[i],trainLabels,splitLabels[i],classifierSettings)
            accuracies[i]= acc
            densities[i] = dens
        averageAccuracy= np.average(accuracies)
        standardDevAccuracy= np.std(accuracies)
        averageDensity=np.average(densities)
        standardDevDensity=np.std(densities)
        return averageAccuracy, standardDevAccuracy,averageDensity,standardDevDensity

    def CrossValid(self,trainData,testData,trainLabels,testLabels, classifierSettings):
        classifier= self.trainClassifier(trainData,trainLabels,classifierSettings)
        density = self.findMarginDensity(trainData,classifier)
        score= classifier.score(testData,testLabels)
        #print(score)
        return density, score

        #checks if drift occurs
    def checkDriftAccuracy(self,batch,labels):
        if batch.size==labels.size:
            score=self.classifier.score(batch,labels)
            if self.avAc-score>self.stdDensThresh*self.stdAc:
                return True
            else:
                return False

        #detects the MD3 drift
    def calculateMD3drift(self,batch):
        batchMD= self.findMarginDensity(batch,self.classifier)
        #print("this is selfdensity" + str(self.density))
        #print("this is batchDensity" + str(batchMD))
        #print("this is selfstandardDeviation"+ str(self.stdDens))
        if np.abs(batchMD-self.density)>self.stdDensThresh*self.stdDens:
            return True
        else:
            return False
    def calculateAccuracy(self,data,labels):
        score = self.classifier.score(data,labels)
        #print("this is accuracy"+ str(score))
        return score
    
    @abstractmethod
    def trainClassifier(self,trainData,trainLabels,classifierSettings):
        pass
    @abstractmethod
    def findMarginDensity(self,classifier,trainData):
        pass
    @abstractmethod
    def train(self,trainData,trainLabels):
        pass
    # @abstractmethod
    # def detectDrift(self,batch):
    #     pass

#initiializes drift detector
#python allows no arguement overloading.
class MD3_V2(MDDriftDetector):
    def __init__(self, kernel="linear",C=1,K=5,theta=2,toobig=False):
        self.kernel=kernel
        self.C=C
        self.K=K
        self.avAc=None
        self.stdAc=None
        self.stdDens=None
        self.classifier=None
        self.density = None
        self.stdDensThresh= theta
        #The ones needed for md3:1
        self.max=self.density
        self.min=self.density
        self.thresh= None

        self.toobig=toobig
        
#needs to be adapted for window stuff.
    def findMarginDensity(self,batch,svm):
        marginDensity=0
        for vect in batch:
            if np.abs(svm.decision_function([vect]))<=1:
                marginDensity= marginDensity+1
        return marginDensity/batch.size
    
    #creates a new classifier then fit it. Return that new Classifier
    #could use wargs for key value pair.

    def trainClassifier(self,data,labels, classifierSettings):
        clf=None
        if self.toobig==True:
            clf = svm.LinearSVC()
            feature_map_nystroem = Nystroem(gamma=.2,
                                random_state=1,
                                n_components=300)
            data_transformed = feature_map_nystroem.fit_transform(data)
        else:
            clf= svm.SVC(kernel=self.kernel,C=self.C)
        clf.fit(data,labels)
        return clf

    #multiclass? linear? etc. retrain once found problem.
    def retrain(self,data,labels,kernel,C,K):
        self.classifier.fit(data,labels)
        density = self.findMarginDensity(data)
        self.max=density
        self.min=density
        avAc,stdAc,stdDens= self.KXVal(data,labels,kernel,C,K)
        self.density=density
        self.avAc=avAc
        self.stdAc=stdAc
        self.stdDens=stdDens
    
    def train(self,data,labels):
        #If we pass stuff yea.
        avAc,stdAc,avDens,stdDens= self.KXVal(data,labels,self.K,self.kernel,self.C)
        self.avAc=avAc
        self.stdAc=stdAc
        self.stdDens=stdDens
        classifier= self.trainClassifier(data,labels,(self.kernel,self.C))
        self.classifier= classifier
        self.density = avDens
##cross avlidaton, gets the distribution components.
    

class MD3_V1(MDDriftDetector):
    def __init__(self,kernel="linear",C=1,threshold=.075,toobig=False):
        classifier= None
        self.kernel=kernel
        self.C=C
        self.max=None
        self.min=None
        self.threshold = threshold
        self.thresh= None
        self.toobig=toobig
        
    def findMarginDensity(self,batch,svm):
        marginDensity=0
        for vect in batch:
            if np.abs(svm.decision_function([vect]))<=1:
                marginDensity= marginDensity+1
        return marginDensity/batch.size
    
    def trainClassifier(self,data,labels, classifierSettings):
        if self.toobig==True:
            clf= svm.LinearSVC(C=self.C)
            feature_map_nystroem = Nystroem(gamma=.2,
                                random_state=1,
                                n_components=300)
            data_transformed = feature_map_nystroem.fit_transform(data)
        else:
            clf= svm.SVC(kernel=self.kernel,C=self.C)
        clf.fit(data,labels)
        return clf
    
    def train(self,data,labels):
        classifier= self.trainClassifier(data,labels,(self.kernel,self.C))
        self.classifier=classifier
        density=self.findMarginDensity(data,classifier)
        self.density=density
        self.max= density
        self.min= density
        self.thresh= self.density * self.threshold

        #threshold=0.075

    #1st version md3
    def calculateMD3drift(self,batch):
        md2=self.findMarginDensity(batch,self.classifier)
        #print(md2)
        if md2>self.max:
            self.max=md2
        if md2<self.min:
            self.min= md2
        if (self.max-self.min)>self.thresh:
            return True
        else: 
            return False



#this is for batch, not sliding window. sliding window forgetting factor sus
#here supposed to query an oracle callback function.

        

class MD3_X(MDDriftDetector):
        
    def __init__(self,estimator,theta=2,K=5,n_estimators=20,max_features=.5,bootstrap=False,bootstrap_features=True):
        #scikit learn parameters
        self.K=K
        self.estimator=estimator
        self.n_estimators=n_estimators
        self.max_features= max_features
        self.bootstrap=bootstrap
        self.bootstrap_features=bootstrap_features
        #nona ble
        self.avAc=None
        self.stdAc=None
        self.stdDens=None
        self.classifier = None
        self.density= None
        self.stdDensThresh=theta


    def trainClassifier(self,data,labels,classifierSettings):
        bagging = BaggingClassifier(self.estimator,n_estimators=self.n_estimators,max_samples=float(1)
        ,max_features=self.max_features,bootstrap= self.bootstrap,bootstrap_features=self.bootstrap_features,n_jobs=1)
        bagging.fit(data,labels)
        return bagging


    def findMarginDensity(self,batch,classifier):
        marginDensity=0
        probabilitiesEach= classifier.predict_proba(batch)
        for vect in probabilitiesEach:
            #print(np.abs(vect[0]-(1-vect[0])),flush=True)
            if np.abs(vect[0]-(1-vect[0]))<.5:
                marginDensity= marginDensity+1
        #print(marginDensity/batch.size)
        return marginDensity/batch.size
    
    def train(self,data,labels):
        avAc,stdAc,avDens,stdDens= self.KXVal(data,labels,self.K,self.estimator,self.n_estimators,self.max_features,self.bootstrap,self.bootstrap_features)
        self.avAc=avAc
        self.stdAc=stdAc
        self.stdDens=stdDens
        self.classifier = self.trainClassifier(data,labels,(self.estimator,self.n_estimators,self.max_features,self.bootstrap,self.bootstrap_features))
        self.density= avDens
        


class FMD(MD3_X):
    
    def membershipFun(self,x):
        membership=0
        confidence = np.abs(x-(1-x))
        if confidence <= .5:
             membership = (np.cos(np.pi*confidence)+1)/2
        return membership

    def findMarginDensity(self,batch,classifier):
        probabilitiesEach = classifier.predict_proba(batch)
        fmd=0
        for vect in probabilitiesEach:
            membership = self.membershipFun(vect[0])
            fmd=fmd+ membership
        return fmd/batch.size




## This is the Pre-processing: Scaling and encoding.


def getCatColumns(pandaFrame):
    names = pandaFrame.select_dtypes(exclude=np.number).columns.tolist()
    return names


def checkandConvertPandas(data):
    if not isinstance(data,pd.DataFrame):
        df= pd.DataFrame(data)
        return df
    else:
        return data



def encodeAndScale(trainData,testBatch,trainLabels,encoder,scalerType):
    le = preprocessing.LabelEncoder()
    labels= le.fit_transform(trainLabels)
    columns = getCatColumns(trainData)
    trainData2=trainData.copy()
    testBatch2=testBatch.copy()
    scaler = None
    if scalerType == "minmax":
        scaler= MinMaxScaler()
    else:
        print("margin density detection doesn't work well without scalers")
        scaler= MinMaxScaler()

    if encoder=="ord":
        ord= preprocessing.OrdinalEncoder()
        trainData2[columns[:]]= ord.fit_transform(trainData2[columns[:]])
        trainData2 = scaler.fit_transform(trainData2[trainData2.columns])
        testBatch2[columns[:]] = ord.transform(testBatch2[columns[:]])
        testBatch2 = scaler.transform(testBatch2[testBatch2.columns])
        #print(trainData2,flush=True)
        return trainData2,testBatch2,labels

    elif encoder=="ohe":
        ohe= preprocessing.OneHotEncoder()
        trainNum = trainData2.select_dtypes(include=[np.number])
        trainNum = scaler.fit_transform(trainNum[trainNum.columns])
        encoded = ohe.fit_transform(trainData2[columns[:]]).toarray()
        trainData2 = np.column_stack((trainNum, encoded))
        
        trainNum2 = testBatch2.select_dtypes(include=[np.number])
        encoded2= ohe.transform(testBatch2[columns[:]]).toarray()
        trainNum2 = scaler.transform(trainNum2[trainNum2.columns])
        testBatch2 = np.column_stack((trainNum2, encoded2))
        #print(testBatch2,flush=True)
        return trainData2,testBatch2,labels

    elif encoder == "targ":
        targ = TargetEncoder(cols=columns, smoothing=0, return_df=False)
        trainData2[columns[:]] = targ.fit_transform(trainData2[columns[:]], labels)
        trainData2 = scaler.fit_transform(trainData2[trainData2.columns])
        testBatch2[columns[:]] = targ.transform(testBatch2[columns[:]])
        testBatch2 = scaler.transform(testBatch2[testBatch2.columns])
        #print(trainData2,flush=True)
        return trainData2,testBatch2,labels
    else:
        print("Not a valid encoder")
        return trainData2,testBatch2,labels



# The functions. One for each drift. The last function takes in a drift detector, which allows tuning.
#The drift detectors and their function.

def findMD3_V1Drift(ref_data,test_data,label,encoder,scaler):
    md3V1= MD3_V1()
    return findDriftWithDetector(md3V1,ref_data,test_data,label,encoder,scaler)

def findMD3_V2Drift(ref_data,test_data,label,encoder,scaler):
    md3v2= MD3_V2()
    return findDriftWithDetector(md3v2,ref_data,test_data,label,encoder,scaler)

def findMD3_XDrift(ref_data,test_data,label,encoder,scaler):
    md3x = MD3_X(DecisionTreeClassifier())
    return findDriftWithDetector(md3x,ref_data,test_data,label,encoder,scaler)

def findFMDDrift(ref_data,test_data,label,encoder,scaler):
    fmd= FMD(DecisionTreeClassifier())
    return findDriftWithDetector(fmd,ref_data,test_data,label,encoder,scaler)


def findDriftWithDetector(detector,ref_data,test_data,label,encoder,scaler):
    trainData = checkandConvertPandas(ref_data)
    testBatches = checkandConvertPandas(test_data)
    trainData, testBatches, trainLabels =encodeAndScale(trainData,testBatches,label,encoder,scaler)
    detector.train(trainData,trainLabels)
    return detector.calculateMD3drift(testBatches)
