# Classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
path="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
headernames=['sepal-length','sepal-width','petal-length','petal-width','Class']
dataset = pd.read_csv(path,names=headernames)
dataset.head()
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
from sklearn.model_selection import train_test_split
x_train=train_test_split(x,y,test_size = 0.30)
x_test=train_test_split(x,y,test_size = 0.30)
y_train=train_test_split(x,y,test_size = 0.30)
y_test=train_test_split(x,y,test_size = 0.30)
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=50)
classifier.fit(x_train,y_train)
RandomForestClassifier(n_estimators=50)
y_pred=classifier.predict(x_test)
from sklearn.metrices import classification_report,confusuin_matrix,accuracy_score
result=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(result)
result1=classification_report(y_test,y_pred)
print("Classification Report:",)
print(result1)
result2=accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
