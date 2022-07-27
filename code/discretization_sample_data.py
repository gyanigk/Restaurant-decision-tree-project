# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:35:41 2022

@author: gyani
https://towardsdatascience.com/discretisation-using-decision-trees-21910483fa4b
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier,export_graphviz

#%%
# discretize age
data = pd.read_csv('train.csv',usecols=['Age','Fare','Survived'])
data.head(10)
data.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data[['Age', 'Fare', 'Survived']],data.Survived , test_size = 0.25)

tree_model = DecisionTreeClassifier(max_depth=2)
tree_model.fit(X_train.Age.to_frame(), X_train.Survived)
X_train['Age_tree']=tree_model.predict_proba(X_train.Age.to_frame())[:,1]
print(X_train.head(10))
print(X_train.Age_tree.unique())

#%%
fig = plt.figure()
fig = X_train.groupby(['Age_tree'])['Survived'].mean().plot()
fig.set_title('Monotonic relationship between discretised Age and target')
fig.set_ylabel('Survived')
#%%
X_train.groupby(['Age_tree'])['Survived'].count().plot.bar()

#%%
bucket_cutoff= pd.concat( [X_train.groupby(['Age_tree'])['Age'].min(),
            X_train.groupby(['Age_tree'])['Age'].max()], axis=1)
print(bucket_cutoff)
#%%
with open("tree_model.txt", "w") as f:
    f = export_graphviz(tree_model, out_file=f)

#%%
score_ls = []     # here I will store the roc auc
score_std_ls = [] # here I will store the standard deviation of the roc_auc
for tree_depth in [1,2,3,4]:
    tree_model = DecisionTreeClassifier(max_depth=tree_depth)
    scores = cross_val_score(tree_model, X_train.Age.to_frame(),
    y_train, cv=3, scoring='roc_auc')
    score_ls.append(np.mean(scores))
    score_std_ls.append(np.std(scores))

temp = pd.concat([pd.Series([1,2,3,4]), pd.Series(score_ls), pd.Series(score_std_ls)], axis=1)
temp.columns = ['depth', 'roc_auc_mean', 'roc_auc_std']
print(temp)

#%%
tree_model = DecisionTreeClassifier(max_depth=2)
tree_model.fit(X_train.Age.to_frame(), X_train.Survived)
X_train['Age_tree'] = tree_model.predict_proba(X_train.Age.to_frame())[:,1]
X_test['Age_tree'] = tree_model.predict_proba(X_test.Age.to_frame())[:,1]

#%%
print(X_train.head(10))
print(X_train.Age_tree.unique())
print(X_test.head(),X_test.Age_tree.unique())


#%%
# discretize Fare

data = pd.read_csv('train.csv',usecols=['Age','Fare','Survived'])
data.head(10)
data.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data[['Age', 'Fare', 'Survived']],data.Survived , test_size = 0.2)

tree_model = DecisionTreeClassifier(max_depth=6)
tree_model.fit(X_train.Fare.to_frame(), X_train.Survived)
X_train['Fare_tree']=tree_model.predict_proba(X_train.Fare.to_frame())[:,1]
print(X_train.head(10))
print(X_train.Fare_tree.unique())

#%%
fig = plt.figure()
fig = X_train.groupby(['Fare_tree'])['Survived'].mean().plot()
fig.set_title('Monotonic relationship between discretised Age and target')
fig.set_ylabel('Survived')
#%%
X_train.groupby(['Fare_tree'])['Survived'].count().plot.bar()

#%%
bucket_cutoff= pd.concat( [X_train.groupby(['Fare_tree'])['Fare'].min(),
            X_train.groupby(['Fare_tree'])['Fare'].max()], axis=1)
print(bucket_cutoff)
#%%
with open("tree_model.txt", "w") as f:
    f = export_graphviz(tree_model, out_file=f)

#%%
data = pd.read_csv('train.csv',usecols=['Age','Fare','Survived'])
data.head(10)
data.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data[['Age', 'Fare', 'Survived']],data.Survived , test_size = 0.3)


score_ls = []     # here I will store the roc auc
score_std_ls = [] # here I will store the standard deviation of the roc_auc
for tree_depth in [1,2,3,4,5,6,7,8]:
    tree_model = DecisionTreeClassifier(max_depth=tree_depth)
    scores = cross_val_score(tree_model, X_train.Fare.to_frame(),
    y_train, cv=3, scoring='roc_auc')
    score_ls.append(np.mean(scores))
    score_std_ls.append(np.std(scores))

temp = pd.concat([pd.Series([1,2,3,4,5,6,7,8]), pd.Series(score_ls), pd.Series(score_std_ls)], axis=1)
temp.columns = ['depth', 'roc_auc_mean', 'roc_auc_std']
print(temp)

#%%
tree_model = DecisionTreeClassifier(max_depth=2)
tree_model.fit(X_train.Fare.to_frame(), X_train.Survived)
X_train['Fare_tree'] = tree_model.predict_proba(X_train.Fare.to_frame())[:,1]
X_test['Fare_tree'] = tree_model.predict_proba(X_test.Fare.to_frame())[:,1]

#%%
print(X_train.head(10))
print(X_train.Fare_tree.unique())
print(X_test.head(),X_test.Fare_tree.unique())