#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as po
import plotly.graph_objs as go
import warnings


# In[3]:


df=pd.read_csv('diamond.csv')
df


# In[4]:


df.describe()


# In[5]:


df=df.dropna() 


# In[6]:


df.drop_duplicates(inplace=True)


# In[222]:


#df


# In[7]:


df.describe()
df.head()


# In[8]:


df.color.unique()


# In[8]:


df['color'] = df['color'].astype("category").cat.codes
df['clarity'] = df['clarity'].astype("category").cat.codes
df['cut'] = df['cut'].astype("category").cat.codes


# In[9]:


df


# In[10]:


plt.hist(df['color'])
plt.title(" Color Distribution in diamond dataset")
plt.ylabel("price")
plt.xlabel("color")
plt.show()


# In[11]:


plt.figure(figsize = (5,5))
sns.heatmap(df.corr(),cmap='coolwarm', annot=True)
plt.show()


# In[12]:


df.corr()


# In[13]:


#bar

plotclarity = df.groupby('clarity').price.mean().reset_index()
plotdata = [
    go.Bar(
        x=plotclarity['clarity'],
        y=plotclarity['price'],
        width = [0.3, 0.3,0.3,0.3],
        marker=dict(
        color=['orange', 'green','teal','magenta','blue','pink','grey','yellow'])
    )
]
plotlayout = go.Layout(
        xaxis={"title": "clarity"},
        yaxis={"title": "price"},
        title='Price by clarity',
        
    )
fig = go.Figure(data=plotdata, layout=plotlayout)
po.iplot(fig)


# In[14]:


sns.violinplot(x='carat', y='price', data=df)
plt.xlabel('Carat')
plt.ylabel('price')
plt.show()


# In[16]:


sns.jointplot(x='cut',y='price',data=df,kind='scatter')


# In[17]:


df['color'] = df['color'].astype("category").cat.codes
df['clarity'] = df['clarity'].astype("category").cat.codes
df['cut'] = df['cut'].astype("category").cat.codes


# In[18]:


df


# In[19]:


df.dtypes


# In[20]:


df=(df-df.mean())/df.std()
df=(df-df.min())/(df.max()-df.min())


# In[21]:


df


# In[22]:


y = df['price']
X = df.drop(columns=['price'])


# In[23]:


y


# In[24]:


ratio = 0.20
total_rows = df.shape[0]
test_size = int(total_rows*ratio)


# In[28]:


X_test = X[0:test_size]
X_train = X[test_size:]
y_test = y[0:test_size]
y_train = y[test_size:]


# In[29]:


X_train.shape


# In[30]:


X_test.shape


# In[31]:


y_train.shape


# In[32]:


y_test.shape


# In[33]:


X_T=np.transpose(X_train)
mulx=np.dot(X_T,X_train)
muly=np.dot(X_T,y_train)


# In[34]:


weights=np.dot(np.linalg.inv(mulx),muly)


# In[35]:


#linear
y_pred=np.dot(X_test,weights)
mse = np.mean(np.square(np.subtract(y_test, y_pred)))
print(mse)


# In[36]:


plt.scatter(y_test, y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('Scatter Plot')

plt.show()


# In[43]:


plt.scatter(y_test, y_test, color='blue', label='Actual test data')
plt.plot(y_test, y_pred, color='pink', label='Predictions')
plt.legend()


# #### References 
# 

# https://stackoverflow.com/questions/26414913/normalize-columns-of-a-dataframe

# https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/

# In[ ]:




