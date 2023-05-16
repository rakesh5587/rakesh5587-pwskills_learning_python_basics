#!/usr/bin/env python
# coding: utf-8

# #it is used only for the classification type plblm and can be used for binary or multiclass classification

# In[3]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[7]:


dataset=load_iris()
dataset.DESCR


# In[11]:


x,y=load_iris(return_X_y=True)#so from here we can directly get our dependent and independent features


# In[12]:


x.shape,y.shape


# In[13]:


X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=.3, random_state=1)


# In[18]:


X_train.shape
#as our training data have contineous value so in this case we have to use Gaussian Naive Bayes theorem


# In[19]:


from sklearn.naive_bayes import GaussianNB
gnB_classifier=GaussianNB()


# In[20]:


gnB_classifier.fit(X_train,y_train)


# In[22]:


y_pred=gnB_classifier.predict(X_test)


# In[24]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
confusion_matrix(y_test,y_pred)


# In[25]:


accuracy_score(y_test,y_pred)


# In[27]:


print(classification_report(y_test,y_pred))


# In[ ]:




