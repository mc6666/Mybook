# 迴歸：Scikit-Learn 與矩陣求解的比較

from sklearn import datasets
ds= datasets.load_boston()


print(ds.DESCR)

import pandas as pd

X = pd.DataFrame(ds.data, columns=ds.feature_names)
y = ds.target
X.head(10)

y

X.info()

import numpy as np
X.AGE.astype(np.float32)

X.isnull().sum()

X.isnull().sum().sum()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape, X_test.shape

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

lr.coef_

lr.intercept_

lr.score(X_test, y_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, lr.predict(X_test))

mean_squared_error(y_test, lr.predict(X_test)) ** .5

from sklearn.metrics import r2_score
r2_score(y_test, lr.predict(X_test))

# 二次迴歸

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
X2 = poly.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape, X_test.shape

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

lr.score(X_test, y_test)

len(lr.coef_)

# 簡單運算

import numpy as np

A = np.array([[2,4],
              [6,2]])

B = np.array([[18],
              [34]])

C = np.linalg.solve(A, B)

print(C)

np.linalg.inv(A) @ B

np.linalg.inv(A.T @ A) @ A.T @ B

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection="3d")

X=A
y=B

X1=X[:, 0].reshape(X.shape[0])
X2=X[:, 1].reshape(X.shape[0])
ax.scatter3D(X1, X2, y, cmap='hsv', marker= 'o', s = [160,160])

X1=np.linspace(2,8,50)
X2=np.linspace(2,8,50)
x_surf, y_surf = np.meshgrid(X1, X2)
z_surf= x_surf *5 +  y_surf * 2 
from matplotlib import cm
ax.plot_surface(x_surf, y_surf, z_surf, cmap=cm.hot)    # plot a 3d surface plot
plt.show()


# Boston by matrix

from sklearn import datasets
ds= datasets.load_boston()

import pandas as pd

X = pd.DataFrame(ds.data, columns=ds.feature_names)
y = ds.target
X.head(10)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

b=np.ones((X_train.shape[0], 1))
b.shape

X_train=np.hstack((X_train, b))

# np.linalg.inv(A.T @ A) @ A.T @ B
W = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
W

X_test.shape, W.shape, y_test.shape

b=np.ones((X_test.shape[0], 1))
b.shape

X_test=np.hstack((X_test, b))

SSE = ((X_test @ W - y_test ) ** 2).sum() 
MSE = SSE / y_test.shape[0]
MSE

RMSE = MSE ** (1/2)
RMSE

# R^2 公式
https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score

y_mean = y_test.ravel().mean()
SST = ((y_test - y_mean) ** 2).sum()
R2 = 1 - (SSE / SST)
R2

