"""
Created on Sun Sep  2 21:11:49 2018
@author: hbp16
"""
import numpy as np
import pandas as pd
df = pd.read_csv('/home/hbp16/DRIVES/Drive E/DataCamp/DengAI/dengue_features_train.csv')
label = pd.read_csv('/home/hbp16/DRIVES/Drive E/DataCamp/DengAI/dengue_labels_train.csv')

df['total_cases'] = label.iloc[:,[3]]
df.iloc[:,[1]].describe()

#df.iloc[:,4:24] = df.iloc[:,4:24].fillna(df.iloc[:,4:24].mean(),inplace=True)

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values="NaN",strategy="mean",axis=0)
imp= imp.fit(df.iloc[:,4:24])
df.iloc[:,4:24] = imp.transform(df.iloc[:,4:24])

#visualize data
import matplotlib.pyplot as plt
plt.scatter(df.iloc[:,[1]],df.iloc[:,[24]])
plt.show()

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
df.iloc[:,[1,2]] = sc.fit_transform(df.iloc[:,[1,2]])

#convert non numeric to numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.iloc[:,[0]] = le.fit_transform(df.iloc[:,[0]])

X = df.iloc[:,[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]
Y = df.iloc[:,[24]]

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=4)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr = lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_pred,y_test)


testdf = pd.read_csv('/home/hbp16/DRIVES/Drive E/DataCamp/DengAI/dengue_features_test.csv')

imp = Imputer(missing_values="NaN",strategy="mean",axis=0)
imp= imp.fit(testdf.iloc[:,4:24])
testdf.iloc[:,4:24] = imp.transform(testdf.iloc[:,4:24])

sc = MinMaxScaler(feature_range=(0,1))
testdf.iloc[:,[1,2]] = sc.fit_transform(testdf.iloc[:,[1,2]])

le = LabelEncoder()
testdf.iloc[:,[0]] = le.fit_transform(testdf.iloc[:,[0]])
X_test_pred = testdf.iloc[:,[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]
y_test_pred = lr.predict(X_test_pred)

subdf = pd.read_csv('/home/hbp16/DRIVES/Drive E/DataCamp/DengAI/submission_format.csv')
subdf['total_cases'] = y_test_pred
subdf['total_cases'] = subdf['total_cases'].astype(np.int64)
subdf.to_csv('sub.csv',encoding='utf-8',index=False)










