#!/usr/bin/env python
# coding: utf-8

# In[9]:


#1. Upsampling 
#2. Down sampling


# In[10]:


import numpy as np
import pandas as pd


# In[11]:


np.random.seed(123)#this will avoid variation of random selection of data from data sets 


# In[16]:


n_samples=1000
Class_0_ratio=.9
n_class_0=int(n_samples*Class_0_ratio)
n_class_1=n_samples-n_class_0
n_class_0,n_class_1


# In[17]:


#creating imbalance dataframe 


# In[18]:


class_0=pd.DataFrame({
    'feature_1':np.random.normal(loc=0,scale=1,size=n_class_0),
    'feature_2':np.random.normal(loc=0,scale=1,size=n_class_0),
    "target":[0]*n_class_0
})


# In[34]:


class_0['feature_2'].mean()


# In[30]:


class_1=pd.DataFrame({
    'feature_1':np.random.normal(loc=2,scale=1,size=n_class_1),
    'feature_2':np.random.normal(loc=2,scale=1,size=n_class_1),
    "target":[1]*n_class_1
})


# In[32]:


class_1['feature_2'].meann()


# In[37]:


df=pd.concat([class_0,class_1]).reset_index(drop=True)


# In[38]:


df.target.value_counts()


# In[39]:


df.head()


# In[40]:


#upsampling


# In[43]:


df_minority=df[df['target']==1]


# In[44]:


df_minority


# In[45]:


df_majority=df[df['target']==0]


# In[46]:


df_majority


# In[49]:


from sklearn.utils import resample


# In[50]:


df_minority_upsampled=resample(df_minority,replace=True,
         n_samples=len(df_majority),
         random_state=24)


# In[51]:


df_minority_upsampled


# In[ ]:




