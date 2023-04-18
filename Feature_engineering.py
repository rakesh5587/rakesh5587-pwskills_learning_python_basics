#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns


# In[2]:


df=sns.load_dataset('titanic')


# In[9]:


df.isnull().sum()


# In[10]:


#we will discusss about Imputation technique


# In[11]:


#with the help of Imputation technique we will fill the missing data


# In[12]:


#1. mean imputation 
#this is applied when you are having normally distributed data


# In[13]:


#2.median imputation
# this method is applied when are having right or left skewed dataset
#Because this type of data sets are having some outliers and we have to remove that 


# In[ ]:


#mode imputation technique
#this method is used for categorical values 

