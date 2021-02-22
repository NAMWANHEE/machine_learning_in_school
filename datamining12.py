import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score


digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.3)
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test,y_pred))


