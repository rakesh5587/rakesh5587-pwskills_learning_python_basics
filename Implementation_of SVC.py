#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


#let us create synthetic datpoints
from sklearn.datasets import make_classification


# In[33]:


x,y=make_classification(n_samples=1000,n_features=2,n_classes=2, n_clusters_per_class=2, n_redundant=0)


# In[34]:


x.shape, y.shape


# In[35]:


pd.DataFrame(x)[0]


# In[36]:


import seaborn as sns
plt.figure(figsize=(15,8))
sns.scatterplot(x=pd.DataFrame(x)[0],y=pd.DataFrame(x)[1],hue=y)


# In[40]:


from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test=train_test_split(x,y,test_size=.25,random_state=43)


# In[41]:


from sklearn.svm import SVC
svc=SVC(kernel="linear")


# In[42]:


svc.fit(x_train,y_train)


# In[43]:


svc.coef_


# In[44]:


y_pred=svc.predict(x_test)


# In[46]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(classification_report(y_test,y_pred))


# In[47]:


accuracy_score(y_test,y_pred)


# In[48]:


confusion_matrix(y_test,y_pred)


# In[ ]:




