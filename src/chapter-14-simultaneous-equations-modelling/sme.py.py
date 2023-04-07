
# In[1]:


import pickle
from linearmodels import IV2SLS
import statsmodels.api as sm
import pandas as pd

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'                     'QMF Book/book Ran/data files new/Book4e_data/'

with open(abspath + 'macro.pickle', 'rb') as handle:
    data = pickle.load(handle)


# In[2]:


# 2SLS, specification 1
data = sm.add_constant(data)
ivmod = IV2SLS(dependent = data.inflation,
               exog = data[['const','dprod','dcredit','dmoney']],
               endog = data.rsandp,
               instruments = data[['rterm','dspread']])
res_2sls1 = ivmod.fit(cov_type='unadjusted')
print(res_2sls1)


# In[3]:


# 2SLS, specification 2
ivmod = IV2SLS(dependent = data.rsandp,
               exog = data[['const','dprod','dcredit','dmoney']],
               endog = data.inflation,
               instruments = data[['rterm','dspread']])
res_2sls2 = ivmod.fit(cov_type='unadjusted')
print(res_2sls2)


