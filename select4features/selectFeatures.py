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

params = {
    'uniform': {},
    'manhattan': {'weights':"distance", 'p':1},
    'euclidean': {'weights':"distance", 'p':2}
}

X_train, X_test, y_train, y_test = train_test_split(wine[D].values, np.ravel(wine[[C]]), test_size=0.2, random_state = 42)
resDictST = {}
resDict = {'uniform': {}, 'manhattan': {}, 'euclidean': {}}
selectedDict = {'uniform': {}, 'manhattan': {}, 'euclidean': {}}
for key, v in params.items():
    resDictST[key] = [accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k, **v).fit(X_train, y_train).predict(X_test)) for k in range(1,51)]
    for i in itertools.combinations(range(len(D)), 4):
        resDict[key][tuple(D[y] for y in i)] = [accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k, **params[key]).fit(X_train[:,list(i)], y_train).predict(X_test[:,list(i)])) for k in range(1,51)]

for key in params.keys():
    for k, v in resDict[key].items():
        if sum(v[i] > resDictST[key][i] for i in range(len(resDictST[key]))) > len(resDictST[key])//2:
            selectedDict[key][k] = v

for k, v in selectedDict.items():
    print([k, v])
