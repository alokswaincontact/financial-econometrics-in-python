
# In[1]:


import pickle
import statsmodels.tsa.api as smt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'                     'QMF Book/book Ran/data files new/Book4e_data/'
with open(abspath + 'UKHP.pickle', 'rb') as handle:
    data = pickle.load(handle)


# In[2]:


data_insample = data['1991-02-01':'2015-12-01']
data_insample.tail()


# In[3]:


model = smt.ARIMA(data_insample['dhp'], order=(2,0,0))
res = model.fit()
print(res.summary())

# In[4]:


model = smt.ARIMA(data['dhp'], order=(2,0,0))
res = model.fit()

fig = res.plot_predict('2016-01-01','2018-03-01',dynamic=False)
fig.set_dpi(600)
plt.show()


# In[5]:


fig = res.plot_predict('2016-01-01','2018-03-01',dynamic=True)
fig.set_dpi(600)
plt.show()


# In[6]:


def rmse(pred, target):
    return np.sqrt(((pred - target) ** 2).mean())

data_outsample = data['2016-01-01':'2018-03-01']
pred = res.predict('2016-01-01','2018-03-01',dynamic=False)

stats1 = rmse(pred, data_outsample['dhp'])
print('root mean squared error1: {}'.format(stats1) )

stats2 = sqrt(mean_squared_error(data_outsample['dhp'], pred))
print('root mean squared error2: {}'.format(stats2) )

