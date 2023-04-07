

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'                     'QMF Book/book Ran/data files new/Book4e_data/'
data = pd.read_excel(abspath + 'SandPhedge.xls', index_col=0)

data.head()

# In[2]:


formula = 'Spot ~ Futures'
results = smf.ols(formula, data).fit()
print(results.summary())

# In[3]:


def LogDiff(x):
    x_diff = 100*np.log(x/x.shift(1))
    x_diff = x_diff.dropna()
    return x_diff
    
data = pd.DataFrame({'ret_spot' : LogDiff(data['Spot']),
                    'ret_future':LogDiff(data['Futures'])})
data.head()


# In[4]:


data.describe()



# In[5]:


formula = 'ret_spot ~ ret_future'
results = smf.ols(formula, data).fit()
print(results.summary())

