#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("height-weight.csv")


# In[3]:


df.head()


# In[4]:


x=df[['Weight']]
y=df[['Height']]


# In[5]:


x


# In[6]:


y


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=1)


# In[17]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[18]:


#statdardize tha data for independent variables
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[19]:


x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[20]:


from sklearn.linear_model import LinearRegression
alg=LinearRegression()
alg.fit(x_train,y_train)


# In[22]:


y_predicted=alg.predict(x_test)


# In[23]:


y_predicted,y_test


# In[24]:


alg.intercept_


# In[27]:


alg.coef_


# In[32]:


alg.score(y_predicted,y_test)


# In[33]:


plt.scatter(x,y)


# In[39]:


plt.scatter(x_train,y_train)
plt.plot(x_train,alg.predict(x_train),'r')#red line shows best fit line


# In[35]:


plt.scatter(y_test,y_predicted)


# In[40]:


re=y_predicted-y_test
import seaborn as sns
sns.distplot(re,kde=True)


# In[41]:


#if this is normaly distrbuted then this shows you model accuracy is high


# In[ ]:




