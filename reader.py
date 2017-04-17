from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy.misc import imread
import tensorflow as tf
import pandas as pd
from random import shuffle
import pylab
import numpy as np
import os
import cv2

baseDataset = "/Users/marcostexeira/Downloads/codigoMorcegos/dataset/"
positivePath = 'PositiveRandom/'

classesNames={}
classCounter=[]
c_C=0
X={}

def img2array(image):
    vector = []
    for line in image:
        for column in line:
            vector.append(float(column[0])/255)

    return np.array(vector)

def parseData():
    count=0
    for idx, img in enumerate(os.listdir(positivePath)):
        if img.find('.png') > 0:
            #splitting name
            validImg = img[1:]
            if 'id' not in validImg:
                if validImg.split('_')[0] == 'Carollia':
                    classesName = validImg.split('_')[0] + ' '+ validImg.split('_')[1]
                    audioName   = validImg.split('_')[2].split('M')[0] + ' M'+ validImg.split('I')[0].split('M')[-1]
                    imgName     = 'c'+ validImg.split('_')[2].split('M')[0] + ' M'+ validImg.split('I')[0].split('M')[-1]
                elif not validImg.split('_')[0] == 'Stenodermatinae':
                    classesName = validImg.split('_')[0] + ' '+ validImg.split('_')[1]
                    audioName   = validImg.split('_')[2] + ' '+ validImg.split('I')[0].split('_')[-1]
                    if len(validImg.split('_'))>3:
                        imgName     = 'c'+ validImg.split('_')[2] + ' '+ validImg.split('_')[3]
                else:
                    classesName = validImg.split('_')[0]
                    audioName   = validImg.split('_')[1] + ' '+ validImg.split('I')[0].split('_')[-1]
                    imgName     = 'c'+ validImg.split('_')[1] + ' '+ validImg.split('_')[2]
                
                if not classesName in classesNames.keys() and os.path.isfile(baseDataset + classesName+'/'+audioName+'/Spec/Crop/'+imgName):
                    if count > 0:
                        classCounter.append(c_C)
                    classesNames[classesName] = count
                    #c_C=0
                    count+=1

                if os.path.isfile(baseDataset + classesName+'/'+audioName+'/Spec/Crop/'+imgName):   
                    if not classesNames[classesName] in X.keys():
                        X[classesNames[classesName]] = positivePath + img
                    else:
                        X[classesNames[classesName]] = X[classesNames[classesName]] + ',' + positivePath + img
                    

    X_=[]
    y_=[]

    realClass = 0
    for classVal in range(len(classesNames)):
        arquivos = X[classVal].split(',')
        
        if len(arquivos) > 12:
            for val in arquivos:
                img_ = cv2.imread(val)
                img_ = img2array(img_)
                img_ = img_.astype('float32')

                X_.append(img_)
                y_.append(realClass)
            realClass+=1
    X_      = np.array(X_)
    y_      = np.array(y_)

    
    return X_,y_

