
# In[1]:


import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt
import pickle
import matplotlib.pyplot as plt

abspath = '../../data/'

data = pd.read_excel(abspath + 'currencies.xls',index_col=[0])

def LogDiff(x):
    x_diff = 100*np.log(x/x.shift(1))
    x_diff = x_diff.dropna()
    return x_diff

data = pd.DataFrame({'reur':LogDiff(data['EUR']),
                     'rgbp':LogDiff(data['GBP']),
                     'rjpy':LogDiff(data['JPY'])})

with open(abspath + 'currencies.pickle', 'wb') as handle:
    pickle.dump(data, handle)



# In[2]:


# VAR 
model = smt.VAR(data)
res = model.fit(maxlags=2) 
print(res.summary())


# In[3]:


res = model.select_order(maxlags=10)
print(res.summary())


# In[4]:


model = smt.VAR(data)
res = model.fit(maxlags=2)

#--------------------------------------------------
# Equation reur, Excluded rgbp
resCausality = res.test_causality(causing=['rgbp'],
                         caused=['reur'],
                         kind='wald',signif=0.05 )



# In[5]:


model = smt.VAR(data)
res = model.fit(maxlags=1) 

# Impulse Response Analysis
irf = res.irf(20)
fig = irf.plot()
fig.set_dpi(600)
plt.show()



# In[6]:


# Forecast Error Variance Decomposition (FEVD)
fevd = res.fevd(20)
fig = fevd.plot()
fig.set_dpi(600)
plt.show()


# In[7]:


data1 = data[['rjpy','rgbp','reur']] # reverse the columns

model = smt.VAR(data1)
res = model.fit(maxlags=1) 

# Forecast Error Variance Decomposition (FEVD)
fevd = res.fevd(20)
fig = fevd.plot()
fig.set_dpi(600)
plt.show()

