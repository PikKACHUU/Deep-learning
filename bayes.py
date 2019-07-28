# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


#upload dataset
dataframe=pd.read_csv('/Users/junye.mao/comp219/winequality-red.csv')
#print dataset information
dataframe.info()
#split dataset
X=dataframe.drop("quality",axis=1)
y=dataframe["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
#Training: naive bayes
gaussianNB = GaussianNB()
gaussianNB.fit(X_train, y_train)
#test
answer = gaussianNB.predict_proba(X_test)[:,:]
report = answer>0.5
print(report)
print(gaussianNB.predict(X_test))
#accuracy
acc_gaussianNB=gaussianNB.score(X_test, y_test)
print("Accuracy of naive bayes is : %f" %acc_gaussianNB)
#confusion matrix
y_pred=gaussianNB.predict(X_test)
confmat=confusion_matrix(y_true = y_test, y_pred = y_pred)
print("Confusion matrix of naive bayes is : ")
print(confmat)
#Training: Random Forest 
randomForest=RandomForestClassifier(n_estimators=100)
randomForest.fit(X_train, y_train)
#test
answer = randomForest.predict_proba(X_test)[:,:]
report = answer>0.5
print(report)
print(randomForest.predict(X_test))
#accuracy
acc_randomForest=randomForest.score(X_test, y_test)
print("Accuracy of random forest is : %f" %acc_randomForest)
#confusion matrix
y_pred = randomForest.predict(X_test)
confmat = confusion_matrix(y_true = y_test, y_pred = y_pred)
print("Confusion matrix of random forest is: ")
print(confmat)
