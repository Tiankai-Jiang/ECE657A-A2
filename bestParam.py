import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import *
import matplotlib.pyplot as plt

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

for j in [C, L]:
    scalers = {
        'z-normalization': train_test_split(StandardScaler().fit_transform(wine[D]), np.ravel(wine[[j]]), test_size=0.2, random_state = 42),
        'RobustScaler': train_test_split(RobustScaler().fit_transform(wine[D]), np.ravel(wine[[j]]), test_size=0.2, random_state = 42),
        'MinMaxScaler': train_test_split(MinMaxScaler().fit_transform(wine[D]), np.ravel(wine[[j]]), test_size=0.2, random_state = 42),
        'MaxAbsScaler': train_test_split(MaxAbsScaler().fit_transform(wine[D]), np.ravel(wine[[j]]), test_size=0.2, random_state = 42),
        'QuantileTransformer': train_test_split(QuantileTransformer(random_state=42).fit_transform(wine[D]), np.ravel(wine[[j]]), test_size=0.2, random_state = 42),
        'PowerTransformer': train_test_split(PowerTransformer().fit_transform(wine[D]), np.ravel(wine[[j]]), test_size=0.2, random_state = 42),
        'Normalizer': train_test_split(Normalizer().fit_transform(wine[D]), np.ravel(wine[[j]]), test_size=0.2, random_state = 42),
        'OrdinalEncoder': train_test_split(OrdinalEncoder().fit_transform(wine[D]), np.ravel(wine[[j]]), test_size=0.2, random_state = 42),
        'OneHotEncoder': train_test_split(OneHotEncoder().fit_transform(wine[D]), np.ravel(wine[[j]]), test_size=0.2, random_state = 42),
    }

    for key, v in scalers.items():
        plt.plot(range(1,51), [accuracy_score(v[3], KNeighborsClassifier(n_neighbors=k, weights = 'distance', p = 1).fit(v[0], v[2]).predict(v[1])) for k in range(1,51)], label=key)
    plt.legend()
    plt.show()