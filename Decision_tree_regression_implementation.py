#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn import datasets
dataset=datasets.fetch_california_housing()


# In[3]:


dataset.DESCR


# In[4]:


dataset.keys()


# In[5]:


df=pd.DataFrame(dataset.data, columns=dataset.feature_names)


# In[6]:


df.head()


# In[7]:


df['target']=dataset.target


# In[8]:


df.head()
df.shape


# In[9]:


#as the dataset is too big so it will take so much time while performimg hyperparameter tuning
df=df.sample(frac=.25)
df.shape


# In[10]:


x=df.drop('target',axis=1)
y=df.target


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=.25,random_state=42)


# In[12]:


from sklearn.tree import DecisionTreeRegressor
rgsrr=DecisionTreeRegressor()


# In[13]:


rgsrr.fit(X_train,y_train)


# In[14]:


y_pred=rgsrr.predict(X_test)
y_pred.shape


# In[15]:


#so in regression prblm we can not calculate confusion matrix,accuacy report,classification report


# In[16]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
# print(confusion_matrix(y_test,y_pred))
# print(accuracy_score(y_test,y_pred))
# print(classification_report(y_test,y_pred))


# In[17]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)


# In[18]:


score
#by default we are not selecting any paramater the otained score is around .50


# In[19]:


#hyperparameter tuning
paramet={
    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter':['best','random'],
    'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12],
    'max_features':['auto','sqrt','log2']
                 
}
regressor=DecisionTreeRegressor()


# In[20]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
model=GridSearchCV(regressor,param_grid=paramet,cv=2,scoring='neg_mean_squared_error')


# In[21]:


model.fit(X_train,y_train)


# In[22]:


model.best_params_


# In[23]:


model.score


# In[24]:


y_pred=model.predict(X_test)


# In[25]:


score=r2_score(y_test,y_pred)


# In[26]:


score


# In[ ]:




