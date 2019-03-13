# Compare Algorithms
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
namesHousing = ['RM', 'LSTAT', 'PTRATIO', 'MEAV']
namesTesting = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
models = []
results = []
names = []
seed = 7
score = None
classScoring = 'accuracy'
regScoring = 'neg_mean_squared_error'

#The data which is a dataframe is set to X, and Y
#colNames is an array of all columns, yCol is the output column if labeled
def set_data(data, colNames, mode):
	if mode == 'manual':
		names = colNames
		dataframe = pandas.read_csv(data, names=names)
		array = dataframe.values
		X = array[:,0:len(names)-1]
		Y = array[:,len(names)-1]
		return X,Y
	elif mode == 'auto':
		dataframe = pandas.read_csv(data)
		Y = dataframe[list(dataframe)[-1]]
		X = dataframe.drop(list(dataframe)[-1], axis = 1)
		print(X,Y)
		return X,Y

def performance_metric(y_true, y_predict):
    score = r2_score(y_true,y_predict)
    return score

prefScoring = make_scorer(performance_metric)

#Prepare and compare various models. addModels is a dictionary object : {'LR1':  LogisticRegression(), ...}
def set_models(addModels, mode):
	if mode == 'classifier' :
		models.append(('LR1', LogisticRegression()))
		models.append(('LDA', LinearDiscriminantAnalysis()))
		models.append(('NB', GaussianNB()))
		models.append(('SVM', SVC()))
		score = classScoring
	elif mode == 'regressor' :
		models.append(('DT', DecisionTreeRegressor()))
		models.append(('RR', Ridge()))
		models.append(('LA', Lasso()))
		models.append(('EN', ElasticNet()))
		score = prefScoring

	for model in addModels:
		models.append((model, addModels[model]))

X,Y = set_data(url, namesHousing, mode='auto')
set_models({}, mode='classifier')

def compare_scores(n_splits_val, models, X, Y):
	for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=score)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)

compare_scores(10, models, X, Y)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
