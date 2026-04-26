import numpy as np
import pandas as pd
from pandas.core.common import random_state
from sklearn.datasets import fetch_openml # openml : every dataset is present in it
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mnist = fetch_openml('mnist_784',version=1)  # 0 to 9 digits my handwritten pixels data hy 28*28 ka

# seperate label and data:
X,Y = mnist['data'],mnist['target']

# train test split:
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=30)

# scaling:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# define machine:
rbm = BernoulliRBM(n_components=256,learning_rate=0.05,n_iter=100,verbose=1) # components:no. of hidden layer neurons
logistic = LogisticRegression(max_iter=1000) # this is the only reg technique which perform classification
dbn_pipeline = Pipeline(steps=[('rbm',rbm),('logistic',logistic)]) # steps: how many models use
dbn_pipeline.fit(X_train_scaled,Y_train) # fit on complete data

dbn_score = dbn_pipeline.score(X_test_scaled,Y_test)
print("DBN classification score = ",dbn_score)


# TASK:

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import BernoulliRBM
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score
#
# data=pd.read_csv("iris_data.csv")
# X= data.iloc[:,[0,1,2,3]].values
# Y= data.iloc[:,[4]].values
# xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=20)
# scaler = StandardScaler()
# xtrain_scale = scaler.fit_transform(xtrain)
# xtest_scale = scaler.transform(xtest)
# rbm = BernoulliRBM(n_components=16, learning_rate=0.01, n_iter=10, verbose=1)
# knn = KNeighborsClassifier(n_neighbors=5)
# dbn_pipeline = Pipeline(steps=[('rbm', rbm), ('knn', knn)])
# dbn_pipeline.fit(xtrain_scale, ytrain)
# y_pred = dbn_pipeline.predict(xtest_scale)
# score = accuracy_score(ytest, y_pred)
# print(" KNN Classification =", score)
#
# # iteration + learning rate change