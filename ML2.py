#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd


# In[8]:


import numpy as np


# In[9]:


from sklearn.metrics import accuracy_score


# In[10]:


from sklearn.svm import SVC


# In[11]:


from sklearn.neighbors import KNeighborsClassifier


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


from sklearn import metrics


# In[14]:


df=pd.read_csv("C://Users//Aditi//Downloads//archive (2)//emails.csv")


# In[15]:



df.head()


# In[16]:


df.isnull().sum()


# In[17]:



X=df.iloc[:,1:3001]
X


# In[26]:


Y=df.iloc[:,-1].values
Y


# In[27]:


train_x,test_x,train_y,test_y= train_test_split(X,Y,test_size=0.25)


# In[28]:


svc = SVC(C=1.0,kernel='rbf',gamma='auto')
svc.fit(train_x,train_y)
y_pred = svc.predict(test_x)
print("Accuracy score for SVC : " , accuracy_score(y_pred,test_y))


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


# In[30]:


knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)


# In[31]:


print(knn.predict(x_test))


# In[32]:


print(knn.score(x_test,y_test))


# In[33]:


print("Confusion matrix: ")
cs = metrics.confusion_matrix(y_pred,test_y)
print(cs)


# In[ ]:




