# 房價迴歸預測

from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ds = datasets.load_boston()

print(ds.DESCR)

import pandas as pd

X = pd.DataFrame(ds.data, columns=ds.feature_names)
X.head()

y = ds.target
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

scaler = preprocessing.StandardScaler()


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression 
clf = LinearRegression()

clf.fit(X_train, y_train)

clf.coef_

clf.intercept_

import numpy as np
np.argsort(abs(clf.coef_))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_squared_error(y_test, clf.predict(X_test))

mean_absolute_error(y_test, clf.predict(X_test))

r2_score(y_test, clf.predict(X_test))

