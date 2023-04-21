#!/usr/bin/env python
# coding: utf-8

# In[1]:


#standardization
import seaborn as sns


# In[2]:


df=sns.load_dataset('tips')


# In[3]:


df.head()


# In[4]:


import numpy as np
mean=np.mean(df['total_bill'])
std=np.std(df['total_bill'])


# In[5]:


mean,std


# In[6]:


normalized_data=[]
for i in list(df['total_bill']):
    z_score=(i-mean)/std
    normalized_data.append(z_score)


# In[11]:


import pandas as pd
df1=pd.DataFrame(normalized_data)


# In[12]:


sns.histplot(df['total_bill'])


# In[10]:


#as after converting or performing fearture scaling distribution will not change much


# In[13]:


sns.histplot(df1)


# In[21]:


from sklearn.preprocessing import StandardScaler
obj=StandardScaler()


# In[28]:


obj.fit([df['total_bill']])


# In[29]:


obj.transform([df['total_bill']])


# In[26]:


#or
pd.DataFrame(obj.fit_transform([df['total_bill']]))


# In[31]:


#normalization---min max scalar


# In[32]:


df=sns.load_dataset('taxis')


# In[34]:


df.head()


# In[35]:


from sklearn.preprocessing import MinMaxScaler
MMS=MinMaxScaler()


# In[42]:


pd.DataFrame(MMS.fit_transform(df[['distance','fare','tip']]), columns=['distance','fare','tip'])


# In[43]:


#unit vector


# In[45]:


from sklearn.preprocessing import normalize
pd.DataFrame(normalize(df[['distance','fare','tip']]))


# In[ ]:




