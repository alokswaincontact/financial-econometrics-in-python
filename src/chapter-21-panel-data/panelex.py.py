
# In[1]:


import pandas as pd
from linearmodels import PooledOLS, PanelOLS, RandomEffects

abspath = '../../data/'
data = pd.read_excel(abspath + 'panelx.xls')

# Note: can not use 'return' as a variable name as it is one of 
# Python keywords
data=data.rename(columns={'return':'ret'})


# In[2]:


# data summary
data['ret'].describe()


# In[3]:


data['beta'].describe()



# In[4]:


# dropna, Note: regression can not be performed if nan exists in the dataset
data = data.dropna()

# set up a multi-index
data = data.set_index(['firm_ident','year'])

# Simple Pooled Regression
model = PooledOLS.from_formula('ret ~ 1 + beta',data)
res = model.fit()
print(res)


# In[5]:


# Panel Regression with Fixed Effects
model = PanelOLS.from_formula('ret ~ 1 + beta + EntityEffects',data)
res = model.fit()
print(res)


# In[6]:


# Panel Regression with Random Effects
model = RandomEffects.from_formula('ret ~ 1 + beta',data)
res = model.fit()
print(res)

