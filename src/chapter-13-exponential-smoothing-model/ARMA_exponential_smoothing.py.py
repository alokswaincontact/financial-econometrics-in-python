

# In[1]:


import pickle
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'                     'QMF Book/book Ran/data files new/Book4e_data/'
with open(abspath + 'UKHP.pickle', 'rb') as handle:
    data = pickle.load(handle)
    
data_insample = data['1991-02-01':'2015-12-01']
data_outsample = data['2016-01-01':'2018-03-01']

model = smt.ExponentialSmoothing(data_insample['dhp']) 
res = model.fit()
pred = res.predict(start='1991-02-01')


# In[2]:


plt.figure(1,dpi=600)
plt.plot(data_insample['dhp'], label='In-the-sample Data')
plt.plot(data_outsample['dhp'], label='Out-of-the-sample Data')
plt.plot(pred, label='Simple Exponential Smoothing')
plt.legend()
plt.show()


# In[3]:


def rmse(pred, target):
    return np.sqrt(((pred - target) ** 2).mean())

stats = rmse(pred,data_insample['dhp'])
print('Optimal smoothing coefficient: {}'.format(res.params['smoothing_level']))
print('root mean squared error: {}'.format(stats) )
print('sum-of-squared residuals: {}'.format(res.sse) )

