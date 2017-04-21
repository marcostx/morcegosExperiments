import sys
import csv
import numpy as np
import scipy
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn import tree

class Classifer(object):
	"""docstring for Classifer"""
	def __init__(self, inp):
		self.inp = inp

	def load_dataset(self,filename):
		f = open(filename)
		x = []
		y = []
		for line in f:
			v = line.rstrip('\n').split(',')
			vf = [float(i) for i in v[:-1]]
			x.append(vf)
			y.append(int(v[-1]))
		return x,y

	def load_dataset_csv(self,filename):
		csvfile = open(filename)
		reader = csv.reader(csvfile, delimiter=',')
		a = []
		b = []
		for row in reader:
			a.append(row)

	def inductorSVM(self,x,y):
	    clf = SVC(probability = True)
	    param_dist = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
	          'kernel': ['rbf'] }
	    random_search = RandomizedSearchCV(clf,param_distributions=param_dist,n_iter = 20,cv=3,scoring='roc_auc')
	    random_search.fit(x,y)
	    print random_search.best_params_
	    return random_search.best_estimator_

	def run(self):
		fname = self.inp
		self.best_classifier=None
		classifiers=[]
		mean_metrics = []

		
		x,y = self.load_dataset(fname)
		x = np.array(x)
		y = np.array(y) 
		n = len(x)
		kf = cross_validation.StratifiedKFold(y,n_folds=10)
		for train, test in kf:
			xtrain, ytrain = x[train], y[train]
			xtest, ytest = x[test], y[test]

			clf = self.inductorSVM(xtrain,ytrain)
			classifiers.append(clf)
	        
			yscore =  clf.predict_proba(xtest)
			ypred  =  clf.predict(xtest)
			mean_metrics.append((roc_auc_score(ytest,yscore[:,1])+accuracy_score(ytest,ypred)+ precision_score(ytest,ypred)+ recall_score(ytest,ypred)+ f1_score(ytest,ypred))/5)
			print "(AUC  %4.3f) (ACC %4.3f) (Precision %4.3f) (Recoil %4.3f) (F1 %4.3f) (Samples %d)"%(roc_auc_score(ytest,yscore[:,1]), accuracy_score(ytest,ypred), precision_score(ytest,ypred), recall_score(ytest,ypred), f1_score(ytest,ypred), len(y))

		self.best_classifier = classifiers[np.argmax(mean_metrics)]

	    
clf = Classifer(sys.argv[1])
clf.run()

joblib.dump(clf.best_classifier, 'model.pkl') 

	
	