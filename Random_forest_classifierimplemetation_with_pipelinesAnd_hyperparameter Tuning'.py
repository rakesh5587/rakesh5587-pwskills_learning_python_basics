#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
dataset=sns.load_dataset('tips')


# In[2]:


dataset.head()


# In[3]:


dataset['time'].unique()


# In[4]:


#now we have to perform feature engineering
dataset.info()


# In[5]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[6]:


dataset['time']=encoder.fit_transform(dataset['time'])


# In[7]:


dataset.head()


# In[8]:


df=dataset.copy()


# In[9]:


df.isna().sum()


# In[10]:


x=df.drop('time',axis=1)
y=df['time']


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=42)


# In[12]:


x_train.shape,y_train.shape


# In[ ]:





# In[13]:


from sklearn.impute import SimpleImputer#handellling missing values
from sklearn.preprocessing import OneHotEncoder#handelling categorical features
#handelling outliers
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline#feature scaling
from sklearn.compose import ColumnTransformer#automating the process


# In[14]:


x_train.info()


# In[15]:


categorical_features=['day','smoker','sex']
numerical_features=['total_bill','tip']


# In[16]:


#faeture engineering autoamtion
# to automate the process we have to create pipelines
#numerical pipe line
num_pipeline=Pipeline(
steps=[
    ('imputer',SimpleImputer(strategy='median')),#missing values
    ('scaler',StandardScaler()),#feature scaling 
]
)
#categorical pipeline
cat_pipeline=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),#handelling missing values
    ('onehotencoder',OneHotEncoder())#categorical features to numerical
])


# In[17]:


# now we have to create wraper to contain the pipelines
preprocessor=ColumnTransformer([
    ('num_pipeline',num_pipeline,numerical_features),
    ('cat_pipeline',cat_pipeline,categorical_features)
])


# In[18]:


x_train=preprocessor.fit_transform(x_train)
x_test=preprocessor.transform(x_test)


# In[19]:


from sklearn.ensemble import RandomForestClassifier


# In[20]:


#automate model training process
models={
    'random forest':RandomForestClassifier()
}


# In[21]:


from sklearn.metrics import accuracy_score


# In[22]:


def evaluate_model(x_train,y_train,x_test,y_test,models):
    report={}
    for i in range(len(models)):
        model=list(models.values())[i]
        model.fit(x_train,y_train)
        
        y_test_pred=model.predict(x_test)
        
        test_model_score=accuracy_score(y_test,y_test_pred)
        report[list(models.keys())[i]]=test_model_score
    return report


# In[23]:


evaluate_model(x_train,y_train,x_test,y_test,models)


# In[ ]:




