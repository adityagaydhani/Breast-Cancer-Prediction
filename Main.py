# ## Prediction of Breast Cancer

# Using the Breast Cancer Wisconsin (Diagnostic) Database, we can create a classifier that can help diagnose patients and predict the likelihood of a breast cancer. A few machine learning techniques will be explored.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import cm as cm
import warnings
import time

def GridSearch(X_train, Y_train, model, param_grid, name):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		kfold = KFold(n_splits=10, random_state=21)
		grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
		grid_result = grid.fit(X_train, Y_train)
		print("%s Best: %f using %s" % (name, grid_result.best_score_, grid_result.best_params_))
		return grid_result.best_params_

def prepareModel(X_train, Y_train, model, name):
	print("Preparing model for {}...".format(name))
	start = time.time()
	model.fit(X_train, Y_train)
	end = time.time()
	print( "Run Time: %f\n" % (end-start))
	return model

def estimateAccuracy(X_test, Y_test, model, name):
	print("Estimating accuracy for {}...".format(name))
	predictions = model.predict(X_test)
	print("Accuracy score %f" % accuracy_score(Y_test, predictions))
	print("Classification report:\n", classification_report(Y_test, predictions))
	print("Confusion matrix:\n", confusion_matrix(Y_test, predictions), "\n")

def initializeModelsForBestParams(best_params):
	models_list_best_params = []
	models_list_best_params.append(('LR', LogisticRegression(**best_params['LR'])))
	models_list_best_params.append(('CART', DecisionTreeClassifier(**best_params['CART'])))
	models_list_best_params.append(('SVM', SVC(**best_params['SVM'])))
	models_list_best_params.append(('NB', GaussianNB(**best_params['NB'])))
	models_list_best_params.append(('KNN', KNeighborsClassifier(**best_params['KNN'])))
	return models_list_best_params

# ## Exploratory analysis
# 
# Load the dataset and do some quick exploratory analysis.

data = pd.read_csv('data.csv', index_col=False)
print("Data shape:", data.shape)

# ## Data visualisation and pre-processing
# 
# First thing to do is to enumerate the diagnosis column such that M = 1, B = 0. Then, I set the ID column to be the index of the dataframe. Afterall, the ID column will not be used for machine learning

data['diagnosis'] = data['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')
data = data.set_index('id')
del data['Unnamed: 32']

# Let's take a look at the number of Benign and Maglinant cases from the dataset. From the output shown below, majority of the cases are benign (0).

print("Data examples for each class:\n", data.groupby('diagnosis').size())

# Next, we visualise the data using density plots to get a sense of the data distribution. From the outputs below, you can see the data shows a general gaussian distribution. 

print("Plotting data distribution...")
data.plot(kind='density', subplots=True, layout=(5,7), sharex=False, legend=False, fontsize=1)
plt.show()

# Correlation matrix
print("Plotting correlation matrix...")
fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 30)
cax = ax1.imshow(data.corr(), interpolation="none", cmap=cmap)
ax1.grid(True)
plt.title('Breast Cancer Attributes Correlation')
# Add colorbar, make sure to specify tick locations to match desired ticklabels
fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
plt.show()

# Finally, we'll split the data into predictor variables and target variable, following by breaking them into train and test sets. We will use 20% of the data as test set.

Y = data['diagnosis'].values
X = data.drop('diagnosis', axis=1).values

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.30, random_state=21)

models_list = []
models_list.append(('LR', LogisticRegression()))
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('SVM', SVC())) 
models_list.append(('NB', GaussianNB()))
models_list.append(('KNN', KNeighborsClassifier()))

num_folds = 10
results = []
names = []

print("Calculating validation accuracies for non-standardized data for all models...")
for name, model in models_list:
    kfold = KFold(n_splits=num_folds, random_state=123)
    start = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    end = time.time()
    results.append(cv_results)
    names.append(name)
    print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))

# Performance comparison
fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Standardize the dataset
pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
                                                                       LogisticRegression())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
                                                                        DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',
                                                                      GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
                                                                       KNeighborsClassifier())])))
                                                                       
results = []
names = []
print("Calculating validation accuracies for standardized data for all models...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits=num_folds, random_state=123)
    for name, model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))
        
fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

## Initialize variables for Grid Search
models_list = []
models_list.append(('LR', LogisticRegression()))
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('SVM', SVC())) 
models_list.append(('NB', GaussianNB()))
models_list.append(('KNN', KNeighborsClassifier()))

# param_grid for LR
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
solver_values = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
param_grid_LR = dict(C=c_values, solver=solver_values)

# param_grid for CART
criterion_values = ['gini', 'entropy']
splitter_values = ['best', 'random']
max_features_values = ['auto', 'sqrt', 'log2', None]
class_weight_values = ['balanced', None]
param_grid_CART = dict(criterion=criterion_values, splitter=splitter_values,
					max_features=max_features_values, class_weight=class_weight_values)

# param_grid for SVM
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid_SVM = dict(C=c_values, kernel=kernel_values)

# param_grid for NB
param_grid_NB = dict()

# param_grid for KNN
n_neighbors_values = [5, 10]
weights_values = ['uniform', 'distance']
algorithm_values = ['auto', 'ball_tree', 'kd_tree', 'brute']
param_grid_KNN = dict(n_neighbors=n_neighbors_values,
					weights=weights_values,
					algorithm=algorithm_values)

param_grid_list = []
param_grid_list.append(param_grid_LR)
param_grid_list.append(param_grid_CART)
param_grid_list.append(param_grid_SVM)
param_grid_list.append(param_grid_NB)
param_grid_list.append(param_grid_KNN)

best_params = dict()

## Perform Grid Search for all algorithms on non-standardized data
print("Performing hyperparameter tuning using grid search on non-standardized data for all models...")
for (name, model), param_grid in zip(models_list, param_grid_list):
	if name == 'SVM': best_params[name] = dict()
	else: best_params[name] = GridSearch(X_train=X_train, Y_train=Y_train, model=model, param_grid = param_grid, name=name)

models_list_best_params = initializeModelsForBestParams(best_params)

# Prepare model and estimate accuracy for all algorithms
print("Preparing model and estimating accuracy using best parameters for non-standardized data...")
for name, model in models_list_best_params:
	prepareModel(X_train=X_train, Y_train=Y_train, model=model, name=name)
	estimateAccuracy(X_test=X_test, Y_test=Y_test, model=model, name=name)

## Perform Grid Search for all algorithms on standardized data
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	scaler = StandardScaler().fit(X_train)
	rescaledX = scaler.transform(X_train)
	scaler = StandardScaler().fit(X_test)
	X_test_scaled = scaler.transform(X_test)

print("Performing hyperparameter tuning using grid search on Standardized data...")
for (name, model), param_grid in zip(models_list, param_grid_list):
	best_params[name] = GridSearch(X_train=rescaledX, Y_train=Y_train, model=model, param_grid = param_grid, name=name)

models_list_best_params = initializeModelsForBestParams(best_params)

# Prepare model and estimate accuracy for all algorithms
print("Preparing model and estimating accuracy using best parameters for standardized data...")
for name, model in models_list_best_params:
	prepareModel(X_train=rescaledX, Y_train=Y_train, model=model, name=name)
	estimateAccuracy(X_test=X_test_scaled, Y_test=Y_test, model=model, name=name)
