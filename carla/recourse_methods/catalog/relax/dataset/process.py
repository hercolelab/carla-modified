# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 00:29:06 2021

@author: Albert
"""
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("Covid_ori.csv")
for i in range(17):
    df[str(i)] = preprocessing.scale(df[str(i)])

df.to_csv("Covid.csv")
