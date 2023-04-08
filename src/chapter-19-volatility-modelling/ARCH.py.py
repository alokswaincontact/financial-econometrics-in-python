

# In[1]:


import pickle
from statsmodels.stats.diagnostic import het_arch
from statsmodels.compat import lzip
import statsmodels.api as sm

abspath = '../../data/'
with open(abspath + 'currencies.pickle', 'rb') as handle:
    data = pickle.load(handle)


# In[2]:


data1 = sm.add_constant(data['rgbp'])
results = sm.OLS(data1['rgbp'],data1['const']).fit()
print(results.summary())


# In[3]:


res = het_arch(results.resid,maxlag=5)
name = ['lm','lmpval','fval','fpval']    
lzip(name,res)   

