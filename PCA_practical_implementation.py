#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets


# In[8]:


x,y=datasets.load_iris(return_X_y=True)


# In[9]:


x.shape,y.shape


# In[10]:


x


# In[11]:


y


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=1)


# In[13]:


#by using PCA you can easily perform dimentionality reduction that will help in case our dataset is having large numbers of features so it will increase the performance of the model and we can also visualize our dataset easily and understand


# In[14]:


#PCA_transformation
pca=PCA(n_components=3)


# In[15]:


x_train=pca.fit_transform(x_train)


# In[16]:


x_train #it transformed the 4 features into 3 features


# In[17]:


#for test dataset
pca.transform(x_test)


# In[19]:


pca.components_#it shows eigen vectors


# In[21]:


pca.explained_variance_ratio_#so it explin about data captured by our PC like PC1,PC2, and PC3


# In[22]:


#so PC1 captured 92 percent of the total data 


# In[ ]:




