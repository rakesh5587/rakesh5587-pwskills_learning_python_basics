#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_classification
#with the help of make classification we can generate classification data according to our requirement
from sklearn.linear_model import LogisticRegression


# In[2]:


x,y=make_classification(n_samples=1000,n_features=10, n_informative=5, n_redundant=5,random_state=1)


# In[3]:


x.shape


# In[4]:


y.shape


# In[5]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)


# In[6]:


#model training hyperparameter tuning
#gridsearchCV


# In[7]:


from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


# In[8]:


parameters={'penalty':('l1','l2','elasticnet'),'C':[1,10,20,30]}


# In[9]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()


# In[10]:


clf=GridSearchCV(LogisticRegression,param_grid=parameters,cv=5)


# In[11]:


#splitting of training data into train and validation data
clf.fit(x_train,y_train)


# In[ ]:




