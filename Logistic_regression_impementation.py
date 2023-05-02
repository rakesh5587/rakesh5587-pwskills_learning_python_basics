#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


from sklearn import datasets
dataset=datasets.load_iris()#it will be in the form of dictionary


# In[9]:


print(dataset.DESCR)


# In[11]:


dataset.keys()# so we can get the keys of the dictionary


# In[14]:


df=pd.DataFrame(dataset.data,columns=dataset.feature_names)


# In[15]:


df.head()


# In[16]:


df['target']=dataset.target


# In[17]:


df.head()


# In[19]:


df_copy=df[df['target']!=2]


# In[21]:


df_copy['target'].unique()#so we have only 2 types of taget values


# In[46]:


#dependent and independent features
x=df_copy.drop('target',axis=1)


# In[47]:


y=df_copy['target']


# In[48]:


x.shape,y.shape


# In[49]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)


# In[50]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=200)# in 100 iteration it could not reach to the global minima 


# In[51]:


model.fit(x_train,y_train)


# In[52]:


y_pred=model.predict(x_test)


# In[53]:


y_pred,y_test


# #confusion matrix , precison, classification report
# 

# In[55]:


#all are present in sklearn metrics library


# In[56]:


from sklearn.metrics import confusion_matrix,precision_score,classification_report
print(confusion_matrix(y_test,y_pred))


# In[57]:


print(precision_score(y_test,y_pred))


# In[58]:


print(classification_report(y_test,y_pred))


# #cross validation

# In[60]:



from sklearn.model_selection import KFold
CV=KFold(n_splits=5)


# In[62]:


#these are the scores for all the cross validation
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,x_train,y_train,scoring="accuracy",cv=CV)


# In[63]:


final_score=np.mean(scores)


# In[ ]:




