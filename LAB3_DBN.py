import numpy as np
import pandas as pd
from pandas.core.common import random_state
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from  sklearn.linear_model import LogisticRegression
mnist=fetch_openml('mnist_784',version=1)
x,y=mnist['data'],mnist['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
rbm=BernoulliRBM(n_components=256,learning_rate=0.01,n_iter=20,verbose=1)
logistic=LogisticRegression(max_iter=1000)
dbn_pipeline=Pipeline(steps=[('rbm',rbm),('logistic',logistic)])
dbn_pipeline.fit(x_train_scaled,y_train)
dbn_score=dbn_pipeline.score(x_test_scaled,y_test)
print("DBN CLASSIFICATION SCORE=",dbn_score)

###################TASK IRIS###########################
# import numpy as np
# import pandas as pd
# from pandas.core.common import random_state
# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import BernoulliRBM
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# data = pd.read_csv('Iris.csv')
# x = data.iloc[:, :-1].values
# y = data.iloc[:, -1].astype("category").cat.codes.values
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)
# rbm = BernoulliRBM(n_components=256, learning_rate=0.01, n_iter=20, verbose=1)
# rf_classifier = RandomForestClassifier(n_estimators=100)
# dbn_pipeline = Pipeline(steps=[('rbm', rbm), ('rf_classifier', rf_classifier)])
# dbn_pipeline.fit(x_train_scaled, y_train)
# dbn_score = dbn_pipeline.score(x_test_scaled, y_test)
# print("DBN CLASSIFICATION SCORE=", dbn_score)
