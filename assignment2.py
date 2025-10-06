import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3.csv")

data.shape

y = data['meal']
X = data.drop(['meal','id', 'DateTime'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) #with 14K observations, we will take 10% to test your model

from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.5, objective='multi:softmax', num_class=len(y.unique()))
modelFit = model.fit(X, y)

pred = modelFit.predict(x_test)
print(f"Accuracy score: {accuracy_score(y_test, pred)*100:.2f}%")

testFile = "https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3test.csv"
dataTest = pd.read_csv(testFile)
X = dataTest.drop(['meal','id', 'DateTime'], axis=1)

pred = modelFit.predict(X).tolist()
