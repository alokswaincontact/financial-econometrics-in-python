
# In[1]:


import pickle
from arch.unitroot import DFGLS, ADF, KPSS, PhillipsPerron

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'                     'QMF Book/book Ran/data files new/Book4e_data/'
with open(abspath + 'UKHP.pickle', 'rb') as handle:
    data = pickle.load(handle)

data['First Difference of HP'] = data['Average House Price'].diff()
data = data.dropna()
data.head()

# In[2]:


# test level
res = ADF(data['Average House Price'], lags=10)
print(res.summary())

# In[3]:


res = ADF(data['First Difference of HP'], lags=10)
print(res.summary())


# In[4]:


res = ADF(data['dhp'], lags=10)
print(res.summary())


# In[5]:


res = DFGLS(data['Average House Price'], max_lags=10)
print(res.summary())

