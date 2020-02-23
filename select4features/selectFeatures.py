import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import itertools

D = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
L = 'quality'
C = 'color'
DL = D + [L]
DC = D + [C]
DLC = DL + [C]

wine_r = pd.read_csv("winequality-red.csv", sep=';')
wine_w = pd.read_csv("winequality-white.csv", sep=';')
wine_w[C]= np.zeros(wine_w.shape[0])
wine_r[C]= np.ones(wine_r.shape[0])
wine = pd.concat([wine_w,wine_r])

wine[D] = StandardScaler().fit_transform(wine[D])

n_neighborslist = list(range(1,51))
X_train, X_test, y_train, y_test = train_test_split(wine[D].values, np.ravel(wine[[L]]), test_size=0.2, random_state = 42)

with open('output.csv','a') as fd:
    for i in itertools.combinations(range(len(D)), 4):
        res = [accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k, weights = "distance", p = 2).fit(X_train[:,list(i)], y_train).predict(X_test[:,list(i)])) for k in n_neighborslist]
        fd.write(','.join([str(x) for x in [D[y] for y in i] + res]) + '\n')
# st = [accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k, weights = "distance", p = 2).fit(X_train, y_train).predict(X_test)) for k in n_neighborslist]
# print(st)