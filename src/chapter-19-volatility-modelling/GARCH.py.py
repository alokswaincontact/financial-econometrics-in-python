
# In[1]:


import pickle
from arch import arch_model

abspath = '../../data/'
with open(abspath + 'currencies.pickle', 'rb') as handle:
    data = pickle.load(handle)  


# In[2]:


# The default set of options produces a model with a constant mean, 
# GARCH(1,1) conditional variance and normal errors.   
am = arch_model(data['rjpy'], vol='GARCH')
res = am.fit()
print(res.summary())   


# In[3]:


# E-GARCH   
am = arch_model(data['rjpy'], vol='EGARCH',o=1)
res = am.fit()
print(res.summary())    



# In[4]:


# GJR-GARCH   
am = arch_model(data['rjpy'], p=1, o=1, q=1, vol='GARCH')
res = am.fit()
print(res.summary())

