# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:16:26 2022

@author: gyani

https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

from sklearn.tree import export_graphviz

from IPython.display import Image
import pydotplus

#%%
data = pd.read_csv('C:/Users/gyani/Documents/Github-cloned/Restaurant-decision-tree-project/Social_Network_Ads.csv')
data.head()

feature_cols = ['Age','EstimatedSalary']
X = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values

X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.25, random_state= 0)

#feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train,y_train)

#prediction
y_pred = classifier.predict(X_test)
#Accuracy
print('Accuracy Score:', metrics.accuracy_score(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)

#%%

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop= X_set[:,0].max()+1, step = 0.01),np.arange(start = X_set[:,1].min()-1, stop= X_set[:,1].max()+1, step = 0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75, cmap = ListedColormap(("red","green")))
# dot_data = StringIO()
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1], c = ListedColormap(("red","green"))(i),label = j)
plt.title("Decision Tree(Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

#%%
from six import StringIO
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_png("DT_without_pruning.png")

#%%
classifier = DecisionTreeClassifier(criterion="entropy", max_depth=3)# Train Decision Tree Classifer
classifier = classifier.fit(X_train,y_train)#Predict the response for test dataset
y_pred = classifier.predict(X_test)# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_png("DT_with_pruning.png")