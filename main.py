#passos para realizar os experimentos

# ler as pastas PositiveRandom e NegativeRandom
# para cada imagem lida, se a imagem existir dentro de dataset, ler o arquivo correspondente
# obter as features do arquivo correspondente usando fft (freq_max, freq_min, duracao, pot_max)
# guardar essas features em um dicionario referenciado pelo nome da pasta
# usar as features para treinar um dfa
# usar as imagens lidas para treinar uma rede convolucional
# execucao : python main.py PositiveRandom

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from pyAudioAnalysis import audioBasicIO,audioFeatureExtraction
from random import shuffle
import numpy as np
import re
import csv
import os	
import sys
import math
import warnings #TODO corrigir future warning
import wave
import scipy.io.wavfile as wavfile
import warnings
warnings.filterwarnings('ignore')

from glob import glob
from os.path import basename, join, exists
mfccRanges = [2,21]

def inductorKNN(x,y):
    clf = KNeighborsClassifier()
    param_dist = {'n_neighbors': range(1,100),'weights' : ['distance','uniform'], 'metric': ['minkowski','manhattan','euclidean','chebyshev']}
    random_search = RandomizedSearchCV(clf,param_distributions=param_dist,n_iter=30,cv=3)
    random_search.fit(x,y)

    print random_search.best_params_
    return random_search.best_estimator_

def featureExtractor(fileName):
	[Fs, x] = audioBasicIO.readAudioFile(fileName)
	Features = audioFeatureExtraction.stFeatureExtraction(x, Fs,0.001 * Fs, 0.0003 * Fs)
	MFCCs = []
	
	for index in range(len(Features)):
		MFCCs.append(float(np.mean(Features[index])))
	return MFCCs


baseDataset = "/Users/marcostexeira/Downloads/codigoMorcegos/dataset/"
positivePath = sys.argv[1]
audioPaths=[]
imgFiles=[]
imgFilesPerClass=[]
indexDict={}
classesNames={}
audioFileNames={}
classesIds = {}
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
				classesNames[classesName] = count
				count+=1
			
			if os.path.isfile(baseDataset + classesName+'/'+audioName+'/Spec/Crop/'+imgName):			
				if not baseDataset + classesName+'/'+audioName+'/' in audioFileNames.keys():
					audioFileNames[baseDataset + classesName+'/'+audioName+'/'] = imgName[1:].replace('png','WAV')
				else:
					audioFileNames[baseDataset + classesName+'/'+audioName+'/'] = audioFileNames[baseDataset + classesName+'/'+audioName+'/'] + ', ' +imgName[1:].replace('png','WAV')

featuresExtracted={}
classId = {}
X={}

for value in audioFileNames:
	arquivos = audioFileNames[value].split(', ')
	for arquivo in arquivos:
		#signal_enconter(value + arquivo, 55000, 120000, 200)
		#featuresExtracted[value+arquivo] = featureExtractor(value + arquivo)
		if(classesNames[value.split('/')[6]] in X.keys()):
			X[classesNames[value.split('/')[6]]].append(featureExtractor(value + arquivo))
		else:
			singleList = []
			singleList.append(featureExtractor(value + arquivo))
			X[classesNames[value.split('/')[6]]] = singleList

# training a classifier
count=0
classifiers = []
acc_vals	= []
X_=[]
y_multiclass= []

bestClassifiers = []
for i in range(len(classesNames)):
	if len(X[i]) > 8:
		for val in X[i]:
			X_.append(val)
			y_multiclass.append(i)

X_ = np.array(X_)
y_multiclass = np.array(y_multiclass)

skf = StratifiedKFold(n_splits=10, shuffle=True)

for train_index, test_index in skf.split(X_, y_multiclass):
	shuffle(test_index)
	shuffle(train_index)

	X_train, X_test = X_[train_index], X_[test_index]
	y_train, y_test = y_multiclass[train_index], y_multiclass[test_index]

	#clf = inductorKNN(X_train, y_train)
	clf = OneVsRestClassifier(LinearDiscriminantAnalysis()).fit(X_train, y_train)
	
	pred = clf.predict(X_test)
	acc_vals.append(accuracy_score(y_test, pred ))
	
	print("accuracy : ", accuracy_score(y_test, pred ) )
	print("precision : ", precision_score(y_test, pred, average='weighted' ) )
	print("recall : ", recall_score(y_test, pred,average='weighted' ) )
	print("f1 : ", f1_score(y_test, pred,average='weighted' ) )
	print("\n")


print(np.sum(acc_vals)/10)