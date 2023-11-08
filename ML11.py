#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import warnings

warnings.filterwarnings("ignore")


# In[5]:


data=pd.read_csv("C://Users//Aditi//Downloads//uber.csv")


# In[7]:


df=data.copy()


# In[8]:


df.head()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.corr()


# In[14]:


df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])


# In[15]:


df.info()


# In[16]:


df.isnull().sum()


# In[17]:


df.dropna(inplace=True)


# In[18]:


plt.boxplot(df["fare_amount"])


# In[19]:


q_low=df["fare_amount"].quantile(0.01)
q_high=df["fare_amount"].quantile(0.99)
df=df[(df["fare_amount"]>q_low) & (df["fare_amount"]< q_high)]


# In[20]:


df.isnull().sum()


# In[21]:


from sklearn.model_selection import train_test_split


# In[23]:


x=df.drop("fare_amount",axis=1)

y=df["fare_amount"]


# In[25]:


x["pickup_datetime"]= pd.to_numeric(pd.to_datetime(x["pickup_datetime"]))
x=x.loc[:,x.columns.str.contains("^Unnamed")]


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[28]:


from sklearn.linear_model import LinearRegression


# In[29]:


lrmodel = LinearRegression()
lrmodel.fit(x_train,y_train)


# In[30]:


predict=lrmodel.predict(x_test)


# In[31]:


from sklearn.metrics import mean_squared_error
lrmodelrmse=np.sqrt(mean_squared_error(predict,y_test))
print("RMSE error for the model is ",lrmodelrmse)


# In[32]:


from sklearn.ensemble import RandomForestRegressor


# In[33]:


rfrmodel=RandomForestRegressor(n_estimators=100,random_state=101)


# In[34]:


rfrmodel.fit(x_train,y_train)
rfrmodel_pred = rfrmodel.predict(x_test)


# In[35]:


rfrmodel_rmse=np.sqrt(mean_squared_error(rfrmodel_pred,y_test))
print("RMSE value for the forest is ",rfrmodel_rmse)


# In[ ]:




