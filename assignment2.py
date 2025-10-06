import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3.csv")

# Convert 'DateTime' to datetime objects and extract features
data['DateTime'] = pd.to_datetime(data['DateTime'])

data = data.drop(['DateTime', 'id'], axis=1)


y = data['meal']
X = data.drop('meal', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.5, objective='multi:softmax', num_class=len(y.unique()))
modelFit = model.fit(x_train, y_train)

pred_y = modelFit.predict(x_test)
print(f"Accuracy score: {accuracy_score(y_test, pred_y)*100:.2f}%")

testFile = "https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3test.csv"
dataTest = pd.read_csv(testFile)

# Apply the same feature engineering to the test data
dataTest['DateTime'] = pd.to_datetime(dataTest['DateTime'])
dataTest = dataTest.drop(['DateTime', 'id', 'meal'], axis=1) # Drop 'meal' from test data
X_test_final = dataTest
pred = modelFit.predict(X_test_final).tolist()
