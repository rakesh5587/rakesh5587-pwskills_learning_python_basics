#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from sklearn.preprocessing import OneHotEncoder


# In[4]:


df=pd.DataFrame({'color':['red','blue','green','green','red','blue']})


# In[5]:


df


# In[6]:


#create an instance of onehotencoder


# In[10]:


encoder=OneHotEncoder()
encoded=encoder.fit_transform(df[['color']]).toarray()


# In[12]:


pd.DataFrame(encoded, columns=encoder.get_feature_names_out())


# In[13]:


#label encoder


# In[14]:


# is a method in which we assign unique value to the each gategorical variable


# In[18]:


from sklearn.preprocessing import LabelEncoder


# In[21]:


encode=LabelEncoder()
encode.fit_transform(df[['color']])


# In[23]:


#in label encoding machine may get confused about their ranking so to resolve this we perform ordinal encoding


# In[25]:


#ordinal encoding--used to encode categorical data that have intrinsic order or ranking


# In[26]:


from sklearn.preprocessing import OrdinalEncoder
orden=OrdinalEncoder()


# In[28]:


df=pd.DataFrame({
    'level':['small','medium','large','medium','small', 'large']
})


# In[29]:


df


# In[33]:


#create an instance of ordinal variable 
orden=OrdinalEncoder(categories=[['small','medium','large']])#in categories we have to pass in rank wise 


# In[34]:


orden.fit_transform(df[['level']])


# In[37]:


#target guided ordinal encoding-- in this method encode categorical variables based on the relatonship with the target varibles used when categories are more


# In[40]:


df=pd.DataFrame({'city':['Newyork', 'london', 'paris', 'tokyo', 'Newyork', 'paris'],
             'pric':[200,150,300,250,180,320]})


# In[52]:


mean_price=df.groupby('city')['pric'].mean().to_dict()


# In[53]:


mean_price


# In[54]:


df['df_encoded']=df['city'].map(mean_price)


# In[55]:


df


# In[ ]:




