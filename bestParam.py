import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

ss = StandardScaler().fit_transform(wine[D])
qt = QuantileTransformer().fit_transform(wine[D])
rs = RobustScaler().fit_transform(wine[D])

n_neighborslist = list(range(1,51))
X_train, X_test, y_train, y_test = train_test_split(ss, np.ravel(wine[[L]]), test_size=0.2, random_state = 42)
manhattanSS = [accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k, weights = 'distance', p = 1).fit(X_train, y_train).predict(X_test)) for k in n_neighborslist]

X_train, X_test, y_train, y_test = train_test_split(rs, np.ravel(wine[[L]]), test_size=0.2, random_state = 42)
t = [accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k, weights = 'distance', p = 1).fit(X_train, y_train).predict(X_test)) for k in n_neighborslist]

X_train, X_test, y_train, y_test = train_test_split(qt, np.ravel(wine[[L]]), test_size=0.2, random_state = 42)
t2 = [accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k, weights = 'distance', p = 1).fit(X_train, y_train).predict(X_test)) for k in n_neighborslist]
# for i in range(50):
#     if t[i] > manhattanSS[i]:
#         print(i)
plt.plot(n_neighborslist, t, label="rs")
plt.plot(n_neighborslist, t2, label="qt")
plt.plot(n_neighborslist, manhattanSS, label='ss')
plt.legend()
plt.show()