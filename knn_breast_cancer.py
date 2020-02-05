# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:28:59 2020

@author: renatons
"""
#%%
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
sns.set()
from collections import Counter

#%%
def baseline_calc(classe):
    b1 = Counter(classe.iloc[:, 0])
    mx = max(b1)
    baseline = b1[mx] / sum(b1.values())
    return baseline

#%%
breast_cancer = load_breast_cancer()

base = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)

previsores = base[['mean area', 'mean compactness']]

classe = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)

classe = pd.get_dummies(classe, drop_first=True)

#%%
previsores_treino, previsores_teste, classe_treino, classe_teste = train_test_split(previsores, classe,test_size=0.15, random_state=1)

#%%
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

#%%
knn.fit(previsores_treino, classe_treino)

previsao = knn.predict(previsores_teste)

#%%
sns.scatterplot(x = 'mean area', y = 'mean compactness', hue = 'benign', data = previsores_teste.join(classe_teste, how = 'outer'))

#%%
plt.scatter(previsores_teste['mean area'], previsores_teste['mean compactness'], c=previsao, cmap='coolwarm', alpha=0.7)

#%%
confusion_matrix(classe_teste, previsao)

precisao = accuracy_score(classe_teste, previsao)

baseline = baseline_calc(classe)

