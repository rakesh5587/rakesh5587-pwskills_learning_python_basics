#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from sklearn.datasets import make_classification
x,y=make_classification(n_samples=1000,n_features=3, n_redundant=1,n_classes=2 ,random_state=999)
#n_features means only features not target variable 


# In[10]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=42)


# In[11]:


x_train.shape


# In[12]:


from sklearn.neighbors import KNeighborsClassifier
kn_classifier=KNeighborsClassifier(n_neighbors=5,algorithm="auto")


# In[13]:


kn_classifier.fit(x_train,y_train)


# In[15]:


y_pred=kn_classifier.predict(x_test)


# In[16]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
confusion_matrix(y_test,y_pred)


# In[17]:


accuracy_score(y_test,y_pred)


# In[ ]:




