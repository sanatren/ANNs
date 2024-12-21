import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

df = sns.load_dataset('iris')
print(df['species'].unique())
df['species'] = df['species'].replace({'setosa':1,'versicolor':2,'virginica':3})
print(df.head())
df['species'] = df['species'].astype(int)
df.drop(columns='sepal_width',axis=1)
df.drop(columns='petal_width',axis=1)
x = df[['sepal_length','petal_length',]]
y = df['species']

from sklearn.linear_model import Perceptron
percep = Perceptron()

percep.fit(x,y)
print(percep.coef_)
print(percep.intercept_)

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(x.values,y.values,clf=percep,legend = 2)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Perceptron Decision Regions on Iris')
plt.show()