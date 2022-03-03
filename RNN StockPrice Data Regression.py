# -*- coding: utf-8 -*-
# Created on Wed Oct  6 15:28:00 2021
# Stock forecast using LSTM

# import libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import  Dense,LSTM,Dropout

# read the train and test data
path="F:/aegis/4 ml/dataset/supervised/tensorflow/rnn/ongc/train.csv"
train = pd.read_csv(path)
train.head()
train.tail()
len(train)

path="F:/aegis/4 ml/dataset/supervised/tensorflow/rnn/ongc/test.csv"
test = pd.read_csv(path)
test.head()
len(test)

train.columns
# to predict/forecast the 'close' price

# store the data in an array format
traindata = train.loc[:,'close'].values.reshape(-1,1)
traindata
testdata = test.loc[:,'close'].values.reshape(-1,1)
testdata

# standardise the train and test dataset
mm = preprocessing.MinMaxScaler()
traindata_std = mm.fit_transform(traindata)
testdata_std = mm.fit_transform(testdata)

# check transformed data
traindata_std[0:5]
testdata_std[0:5]

# build the trainx and trainy datasets
# define the lags
lags = 30 # experiment with the lags

# store the x and y data and then convert them into array
trainx = []; trainy = []

for i in range(lags,len(traindata_std)):
    trainx.append(traindata_std[i-lags:i,0])
    trainy.append(traindata_std[i,0])

trainx[0]
trainy[0]

np.__version__

# convert trainx and trainy into array format
trainx = np.array(trainx)
trainy = np.array(trainy)

trainx.shape

# network expects the input data in a 3-D tensor format
trainx = np.reshape(trainx,(trainx.shape[0], trainx.shape[1], 1))
trainx.shape

# build the LSTM network, compile and predict on test data
regr = Sequential()

# LSTM layer 1
regr.add(LSTM(units=32,return_sequences=True, input_shape=(trainx.shape[1],1)))
regr.add(Dropout(0.2))

# LSTM layer 2
regr.add(LSTM(units=32,return_sequences=True))
regr.add(Dropout(0.2))

# LSTM layer 3
regr.add(LSTM(units=32))
regr.add(Dropout(0.2))

# output layer
regr.add(Dense(units=1))

# compile the network
regr.compile(optimizer='adam',loss='mean_squared_error')

# train the data
EPOCHS = 25
regr.fit(trainx,trainy,epochs=EPOCHS,batch_size=5)

# forecast on the test data
totalrec = pd.concat((train['close'],test['close']),0)
totalrec

len(train)
len(test)
len(totalrec)

# # build the testing data for forecast
# = totalrec[len(totalrec) - len(test) - lags]

inputs = totalrec[len(totalrec)-len(test)-lags:].values

# convert the inputs into an array
inputs = inputs.reshape(-1,1)
inputs = mm.fit_transform(inputs)
inputs

# create the testX data for forecast
testx = []

for i in range(lags,len(inputs)):
    testx.append(inputs[i-lags:i,0])
    
testx = np.array(testx)

# reshape into LSTM format
testx = np.reshape(testx, (testx.shape[0],testx.shape[1],1))
testx.shape

# predict/forecast
predy = regr.predict(testx)
predy

# since train and test are in the minmax format, the predictions are also in the minmax format
# need to re-transform the predictions into the actual form

preds = mm.inverse_transform(predy)


predy[0:5]
preds[0:5]

# create dataframe to store actual and predicted values
df = pd.DataFrame({'actual':test.close, 'predicted':preds.reshape(-1)})
print(df)

predy.reshape(-1,1)

# MSE
from sklearn.metrics import mean_squared_error
mse1 = mean_squared_error(df.actual, df.predicted)  

print("Model Errors\n\t MSE = {}\n\tRMSE = {}".format(mse1,np.sqrt(mse1)))

import nltk
nltk.download()
