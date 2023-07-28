# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path =r"C:\Users\vish\Downloads\archive (7).zip"

data = pd.read_csv(path)
print(data)

y=data.iloc[:,4]
x=data.iloc[:,:-1]

state = pd.get_dummies(x['State'],drop_first=True)

x=x.drop('State',axis=1)

x=pd.concat([x,state],axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regessor = LinearRegression()
regessor.fit(x_train, y_train)

y_pred = regessor.predict(x_test)

from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)