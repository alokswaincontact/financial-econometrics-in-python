

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

abspath = '../../data/'
data = pd.read_excel(abspath + 'SandPhedge.xls', index_col=0)


# In[2]:


formula = 'Spot ~ Futures'
hypotheses = 'Futures = 1'
results = smf.ols(formula, data).fit()
f_test = results.f_test(hypotheses)
print(f_test)


# In[3]:


def LogDiff(x):
    x_diff = 100*np.log(x/x.shift(1))
    x_diff = x_diff.dropna()
    return x_diff
    
data = pd.DataFrame({'ret_spot' : LogDiff(data['Spot']),
                    'ret_future':LogDiff(data['Futures'])})


# In[4]:


formula = 'ret_spot ~ ret_future'
hypotheses = 'ret_future = 1'

results = smf.ols(formula, data).fit()
f_test = results.f_test(hypotheses)
print(f_test)

