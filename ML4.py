#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


import numpy as np


# In[3]:


df=pd.read_csv("diabetes.csv")


# In[4]:


df.head()


# In[6]:


df.isnull().sum()


# In[9]:


df.columns[1:-3]


# In[14]:


for column in df.columns[1:-3]:
    df[column].replace(0,np.NaN,inplace = True)
    df[column].fillna(round(df[column].mean(skipna=True)),inplace =True )
    


# In[15]:


df.head()


# In[16]:


X=df.iloc[:,:8]
Y=df.iloc[:,8:]


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier 


# In[22]:


knn=KNeighborsClassifier()


# In[23]:


knn.fit(X_train,Y_train)


# In[27]:


pred =knn.predict(X_test)


# In[29]:


from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
print("Confusion Matrix")
print(confusion_matrix(Y_test,pred))
print("Accuracy Score",accuracy_score(Y_test,pred))
print("Recall Score",recall_score(Y_test,pred))
print("F1 score",f1_score(Y_test,pred))

print("precision_score",precision_score(Y_test,pred))


# In[ ]:





# In[ ]:




