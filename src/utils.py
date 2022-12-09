import numpy as np
import pandas as pd
import requests
import io
from sklearn.model_selection import train_test_split
def get_data():
  url = "https://cdn.jsdelivr.net/gh/ramenfeast/BV-ethnicity-report/data/BV%20Dataset%20copy.csv"
  download = requests.get(url).content
  df = pd.read_csv(io.StringIO(download.decode('utf-8')))
  df = df.drop([394,395,396], axis = 0)
  return df
def get_XY_split():
  X,y = get_XY()
  return train_test_split(X, y,test_size=0.2, random_state=1)
def get_XY():
  df = get_data()
  X = df.iloc[:,:-1]
  y = df["Nugent score"]
  X=X.drop(labels= ['Ethnic Groupa', 'Community groupc '], axis=1)
  #Normalize 16s RNA data
  # have we  tried log scaling the RNA data?
  X.iloc[:,1::]=X.iloc[:,1::]/100
  #Binary y
  y[y<7]=0
  y[y>=7]=1
  return X,y

def ethnicities():
    return get_data().to_numpy()[:,0]
def groups():
    return get_data().to_numpy()[:,1]
def Ftest_XY():
  url = "https://cdn.jsdelivr.net/gh/ramenfeast/BV-ethnicity-report/results/Feature%20selection/Ftestfeatures.csv"
  download = requests.get(url).content
  df = pd.read_csv(io.StringIO(download.decode('utf-8')))
  x,y= get_XY()
  return x[df["Feature Name"]].to_numpy(),y.to_numpy()
def Gini_XY():
  url = "https://cdn.jsdelivr.net/gh/ramenfeast/BV-ethnicity-report/results/Feature%20selection/Ginifeatures.csv"
  download = requests.get(url).content
  df = pd.read_csv(io.StringIO(download.decode('utf-8')))
  x,y= get_XY()
  return x[df["0"]].to_numpy(),y.to_numpy()
def PBcorr_XY():
  url = "https://cdn.jsdelivr.net/gh/ramenfeast/BV-ethnicity-report/results/Feature%20selection/PBcorr.csv"
  download = requests.get(url).content
  df = pd.read_csv(io.StringIO(download.decode('utf-8')))
  x,y= get_XY()
  return x[df["Feature Name"]].to_numpy(),y.to_numpy()
def Pbsig_XY():
  url = "https://cdn.jsdelivr.net/gh/ramenfeast/BV-ethnicity-report/results/Feature%20selection/PBsignificant.csv"
  download = requests.get(url).content
  df = pd.read_csv(io.StringIO(download.decode('utf-8')))
  x,y= get_XY()
  return x[df["Feature Name"]].to_numpy(),y.to_numpy()
def Ttest_XY():
  url = "https://cdn.jsdelivr.net/gh/ramenfeast/BV-ethnicity-report/results/Feature%20selection/Ttestfeatures.csv"
  download = requests.get(url).content
  df = pd.read_csv(io.StringIO(download.decode('utf-8')))
  x,y= get_XY()
  return x[df["0"]].to_numpy(), y.to_numpy()
