# import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. I like it most for plot

from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.cross_validation import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model


data = pd.read_csv("data.csv",header=0)

print data.head(), data.info()

# Removing the Unnamed column 32 
data.drop("Unnamed: 32", axis = 1, inplace = True)

print data.columns

data.drop("id", axis = 1, inplace = True)

features_mean = list(data.columns[1:11])
features_se= list(data.columns[11:20])
features_worst=list(data.columns[21:31])

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
print data.head()

sns.countplot(data['diagnosis'],label="Count")

corr = data[features_mean].corr() # .corr is used for find corelation
plt.figure(figsize=(14,14))
g = sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features_mean, yticklabels= features_mean,
           cmap= 'coolwarm')

g.set_xticklabels(features_mean,rotation=90)
g.set_yticklabels(features_mean,rotation=90)


plt.show()