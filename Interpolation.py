#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


# In[4]:


a=np.array([1,2,3,4,5])
b=np.array([2,4,6,8,10])


# In[5]:


#linear Interpolation


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


plt.scatter(a,b)


# In[8]:


#interpolate the data using linear interpolation


# In[9]:


x_new=np.linspace(1,5,10)


# In[10]:


x_new


# In[14]:


y_new=np.interp(x_new,a,b)


# In[15]:


y_new


# In[16]:


plt.scatter(x_new,y_new)


# In[17]:


#cubic interpolation with scipy


# In[18]:


a=np.array([1,2,3,4,5])
b=np.array([1,8,27,64,125])


# In[19]:


from scipy.interpolate import interp1d


# In[21]:


f=interp1d(a,b,kind="cubic")
x_new=np.linspace(1,5,10)


# In[22]:


y_new=f(x_new)


# In[23]:


plt.scatter(a,b)


# In[24]:


plt.scatter(x_new,y_new)


# In[25]:


#polynomial interpolation


# In[27]:


a=np.array([1,2,3,4,5])
b=np.array([1,4,9,16,25])


# In[28]:


p=np.polyfit(a,b,2)


# In[33]:


x_new=np.linspace(1,5,10)
y_new=np.polyval(p,x_new)


# In[34]:


y_new


# In[35]:


plt.scatter(a,b)


# In[36]:


plt.scatter(x_new,y_new)


# In[ ]:




