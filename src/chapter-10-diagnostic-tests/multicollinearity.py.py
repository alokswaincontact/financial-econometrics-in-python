

# In[1]:


import pickle

abspath = '../../data/'
with open(abspath + 'macro.pickle', 'rb') as handle:
    data = pickle.load(handle)

data = data.dropna() # drop the missing values for some columns


# In[2]:


data = data[['dprod','dcredit','dinflation','dmoney','dspread','rterm']]
data.corr()

