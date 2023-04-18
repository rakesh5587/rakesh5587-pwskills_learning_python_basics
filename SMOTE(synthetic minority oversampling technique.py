#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_classification


# In[34]:


x,y=make_classification(n_samples=1000,n_redundant=0,n_features=2,n_clusters_per_class=1, weights=[0.9])


# In[35]:


x


# In[36]:


x.shape


# In[37]:


y


# In[38]:


y.shape


# In[39]:


import pandas as pd
df1=pd.DataFrame(x,columns=["f1","f2"])


# In[40]:


df1


# In[41]:


df2=pd.DataFrame(y,columns=['target'])


# In[42]:


df2


# In[43]:


df_final=pd.concat([df1,df2], axis=1)


# In[44]:


df_final['target'].value_counts()


# In[45]:


import matplotlib.pyplot as plt


# In[46]:


plt.scatter(df_final.f1,df_final.f2, c=df_final.target)


# In[47]:


from imblearn.over_sampling import SMOTE


# In[48]:


##transform the dataset


# In[50]:


oversample=SMOTE()
x,y=oversample.fit_resample(df_final[['f1','f2']], df_final['target'])


# In[51]:


x.shape


# In[52]:


y.shape


# In[53]:


y.value_counts()


# In[ ]:




