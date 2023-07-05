import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


housedf=pd.read_csv('USA_Housing.csv')


# In[3]:


housedf


# In[4]:


housedf.describe()


# In[5]:


housedf.info()


# In[7]:


housedf.head()


# In[8]:


housedf.columns


# In[9]:


sns.heatmap(housedf.corr(),annot=True)


# In[11]:


X=housedf[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
Y=housedf['Price']


# In[13]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=101) 


# In[14]:


from sklearn.linear_model import LinearRegression 

lm = LinearRegression() 

lm.fit(X_train,y_train) 


# In[15]:


predictions = lm.predict(X_test)  


# In[16]:


plt.scatter(y_test,predictions)


# In[24]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions))) 
