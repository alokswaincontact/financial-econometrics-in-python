
# In[1]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pickle
from datetime import datetime

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'                     'QMF Book/book Ran/data files new/Book4e_data/'
with open(abspath + 'macro.pickle', 'rb') as handle:
    data = pickle.load(handle)

data = data.dropna() # drop the missing values for some columns


# In[2]:


# create January dummy variable
data.index = pd.to_datetime(data.index, format='%Y%m%d')

data['JANDUM'] = np.where(data.index.month == 1, 1, 0)

data['APR00DUM'] = np.where(data.index == datetime(2000,4,1), 1, 0)
data['DEC00DUM'] = np.where(data.index == datetime(2000,12,1), 1, 0)


# In[3]:


# regression
formula = 'ermsoft ~ ersandp + dprod + dcredit + dinflation + dmoney                      + dspread + rterm + APR00DUM + DEC00DUM + JANDUM'
results = smf.ols(formula, data).fit()
print(results.summary())

