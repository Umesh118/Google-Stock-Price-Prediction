#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


# In[4]:


dataset_train = pd.read_csv(r'C:\Users\Umesh\Documents\JupyterNotebook\google stock price prediction\trainset.csv')


# In[5]:


dataset_train.head()


# In[6]:


train_set = dataset_train.iloc[:,1:2].values


# In[7]:


train_set


# In[14]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)


# In[15]:


training_set_scaled


# In[16]:


x_train = []
y_train = []


# In[19]:


for i in range(60,1258):
  x_train.append(training_set_scaled[i-60:i,0])
  y_train.append(training_set_scaled[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)


# In[20]:


#Reshaping
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1)) 


# In[21]:


#Building the RNN
#Import the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[23]:


model = Sequential()
#Adding the first LSTM layer and add dropout regularization
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))
#Adding the second LSTM layer and add dropout regularization
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
#Adding the third LSTM layer and add dropout regularization
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
#Adding the fourth LSTM layer and add dropout regularization
model.add(LSTM(units=50))
model.add(Dropout(0.2))
#Adding the output layer
model.add(Dense(units=1))


# In[24]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[25]:


model.fit(x_train,y_train,epochs=100,batch_size=32)


# In[26]:


# Getting test dataset
test_dataset = pd.read_csv(r'C:\Users\Umesh\Documents\JupyterNotebook\google stock price prediction\testset_1.csv')


# In[28]:


# Getting real stock price of 2020 
real_stock_price = test_dataset.iloc[:,1:2].values


# In[29]:


# predicted stock price of 2020
dataset_total=pd.concat((dataset_train['Open'],test_dataset['Open']),axis=0)
inputs=dataset_total[len(dataset_total)-len(test_dataset)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)


# In[30]:


X_test=[]
for i in range(60,81):
  X_test.append(inputs[i-60:i,0])

X_test=np.array(X_test)  

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))


# In[32]:


predicted_stock_price=model.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)


# In[41]:


#Visualising the result
plt.plot(real_stock_price,color='blue',label='Real Stock Price')
plt.plot(predicted_stock_price,color='red',label='Predicted Stock Price')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Prediction stock price curve')
plt.savefig('Google_Stock_Price_Predictionr.png')
plt.show()


# In[ ]:




