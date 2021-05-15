import numpy as np
import pandas as pd
import seaborn as sns
import plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# init_notebook_mode(connected=True)
import matplotlib.pyplot as plt

data = pd.read_csv('Data/task_b.csv')
data = data.iloc[:, 1:]

print("Dataframe\n", data.head())

# Check if the data is balanced or not

print(data['y'].value_counts())
print("Data is perfectly balanced")
print("*" * 50)
print("Corelation of features with Y")
print(data.corr()['y'])
print("*" * 50)

X = data[['f1', 'f2', 'f3']].values
Y = data['y'].values
print("X shape", X.shape)
print("Y shape", Y.shape)

sns.lmplot('f1', 'f2', data, hue='y', fit_reg=False)

# * As part of this experiment we will observe how linear models work in case of data having feautres with different
# variance * from the output of the above cells we can observe that var(F2)>>var(F1)>>Var(F3)
#
# > Experiment 1:
#     1. We will apply Logistic regression(SGDClassifier with logloss) on 'data' and check the feature importance
#     2. we will apply SVM(SGDClassifier with hinge) on 'data' and check the feature importance
#
# > Experiment 2:
#     1. we will apply Logistic regression(SGDClassifier with logloss) on 'data' after standardization
#        i.e standardization(data, column wise): (column-mean(column))/std(column) and check the feature importance
#     2. we will apply SVM(SGDClassifier with hinge) on 'data' after standardization
#        i.e standardization(data, column wise): (column-mean(column))/std(column) and check the feature importance


# Experiment 1 Before Standardization run Logistic Regression and SVM

# Logistic Regression
logistic_clf = LogisticRegression(penalty='l2')
logistic_clf.fit(X, Y)
# accuracy of Logistic Regression
score = logistic_clf.score(X, Y)

print("accuracy Logistic Regression without standardization ", score)
# Feature importance
print("Feature importance Logistic Regression without standardization ")

classImportance = logistic_clf.coef_[0]
for id, coeff_ in enumerate(classImportance):
    print(f"f{id + 1} coefficient {abs(classImportance[id])} ")

# SVM classifier
svmLinear = LinearSVC(penalty='l2', loss='hinge', max_iter=100000, tol=10e-5)
svmLinear.fit(X, Y)

# accuracy of Logistic Regression
score = svmLinear.score(X, Y)
print("accuracy svm classifier without standardization ", score)

svcImportance = svmLinear.coef_[0]
print("feature importance for SVM without standardization")
for id, coeff_ in enumerate(svcImportance):
    print(f"f{id + 1} coefficient {abs(svcImportance[id])} ")

# Experiment 1 Result :
#
# 1. Logistic regression fits quiet well and accuracy is also good
# 2. SVM linear failed to converge with more than 100000 iteration with hinge loss <br/>
#  Highly variance nature of data is affecting the classifier behavior


# Experiment 2 Feature Standardization on linear svm and logistic

# Scale the feature to zero mean Unit variance
scalar = StandardScaler()
X = scalar.fit_transform(X)

# Logistic Regression on Standardized data
logistic_clf_standard = LogisticRegression(penalty='l2')

logistic_clf_standard.fit(X, Y)
# accuracy of Logistic Regression
score_stanrard = logistic_clf_standard.score(X, Y)
print("accuracy Logistic Regression with standardized feature", score_stanrard)

# Feature importance
classImportance_standard = logistic_clf_standard.coef_[0]
print("classs importance on standardized data")
for id, coeff_ in enumerate(classImportance_standard):
    print(f"f{id + 1} coefficient {abs(classImportance_standard[id])} ")

# Run Same Experiment on SVM
# SVM classfier
svmLinear_standard = LinearSVC(penalty='l2', loss='hinge', max_iter=500, tol=10e-5)
svmLinear_standard.fit(X, Y)
score_svm_standard = svmLinear_standard.score(X, Y)
print(f"Accuracy of SVM linear classfier on standardized dataset {score_svm_standard}")

svcImportance_standard = svmLinear_standard.coef_[0]
print("feature importance for SVM linear with standardized feature")
for id, coeff_ in enumerate(svcImportance_standard):
    print(f"f{id + 1} coefficient {abs(svcImportance_standard[id])} ")

# Experiment 2 Result :
#
# 1. Logistic regression fits quiet well and accuracy is also good with standardized feature
# 2. SVM linear  converge with in  500 max iteration with hinge loss <br/>
# 3. After Standardization SVM Linear converged super fast and accuracy is also increased to 0.92

# 1. Logistic regression fits quiet well and accuracy is also good with standardized feature <br/>
# or without standardization, feature importance is also intact for logistic regression
# 2. SVM linear seems highly sensitive to variance of features in dataset so standardization
# helps to overcome high variance nature of dataset, and helps to improve the classfier
# 3. For SVM feature importance changed after standardization from (f3 > f2 > f1) to (f3 > f1 > f2)


# Final Conclusion

# It is good practice to use some form of standardization before applying logistic or linear svm model
