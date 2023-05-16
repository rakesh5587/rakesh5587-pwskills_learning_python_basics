#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_iris


# In[4]:


dataset=load_iris()
dataset


# In[5]:


dataset.keys()


# In[9]:


df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['target']=dataset.target
df.head()


# In[11]:


x=df.drop('target',axis=1)
y=df.target


# In[12]:


x.shape,y.shape


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=.2,random_state=42)


# In[21]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
#we can also select the criterion according to us 


# In[22]:


classifier.fit(X_train,y_train)


# In[20]:


#now we can also represent our decision tree
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(classifier,filled=True)


# #post pruning 
# #in this technique first we will split our decision tree to leaf node but our dataset got affected by overfitting to avoid overfittig we do perform post pruning where we will cut down the unnecessary levels

# In[25]:


classifier=DecisionTreeClassifier(criterion="entropy",max_depth=2)
classifier.fit(X_train,y_train)


# In[26]:


tree.plot_tree(classifier,filled=True)


# In[28]:


y_pred=classifier.predict(X_test)
y_pred


# In[30]:


#AS for linear regression we have calculated R2score and adjusted r2score to compare the accuracy 
#incase of logistic regression for finding performance of model we will calclate confusion matrix,accuracy score and classification report


# In[31]:


from sklearn.metrics import accuracy_score,classification_report
score=accuracy_score(y_test,y_pred)
print(score)
print(classification_report(y_test,y_pred))


# Decision tree prepruning and hyperparameter tuning 

# In[32]:


import warnings
warnings.filterwarnings('ignore')


# In[33]:


parameters={
    'criterion':['gini','entropy','log_loss'],
    'splitter':['best','random'],
    'max_depth':[1,2,3,4,5],
    'max_features':['auto','sqrt','log2']
}


# In[34]:


from sklearn.model_selection import GridSearchCV
model=DecisionTreeClassifier()


# In[35]:


clf=GridSearchCV(model,param_grid=parameters,cv=5,scoring='accuracy')


# In[36]:


clf.fit(X_train,y_train)


# In[37]:


clf.best_params_


# In[38]:


# you can predict the for the test dataset with the clf or you can create decision treee classifir using obtained parameters


# In[39]:


y_pred=clf.predict(X_test)


# In[40]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
confusion_matrix(y_test,y_pred)


# In[41]:


accuracy_score(y_test,y_pred)


# In[43]:


print(classification_report(y_test,y_pred))


# In[ ]:




