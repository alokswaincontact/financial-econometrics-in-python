
# In[1]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from arch.unitroot import DFGLS

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD'                 '/QMF Book/book Ran/data files new/Book4e_data/'
data = pd.read_excel(abspath + 'SandPhedge.xls', index_col=0)

data['lspot'] = data['Spot'].apply(lambda x : np.log(x)) 
data['lfuture'] = data['Futures'].apply(lambda x : np.log(x)) 

formula = 'lspot ~ lfuture'
results = smf.ols(formula, data).fit()

residuals = results.resid
lspot_fit = results.fittedvalues



# In[2]:


fig = plt.figure(1, dpi=600)
ax1 = fig.add_subplot(111)
ax1.plot(lspot_fit, label='Linear Prediction')
ax1.plot(data['lspot'], label='lspot')
ax1.set_xlabel('Date') 
ax1.legend(loc=0)  

ax2 = plt.twinx()
ax2.set_ylabel('Residuals')
ax2.plot(residuals, label='Residuals')
ax2.legend(loc=0) 

plt.grid(True)
plt.show()

# In[3]:


res = DFGLS(residuals, max_lags=12)
print(res.summary())


# In[4]:


# Error Correction Model
# specification 1: rspot rfutures L.resid
def LogDiff(x):
    x_diff = 100*np.log(x/x.shift(1))
    x_diff = x_diff.dropna()
    return x_diff

data['rspot'] = LogDiff(data['Spot'])
data['rfuture'] = LogDiff(data['Futures'])
data['lresid'] = residuals.shift(1)
formula = 'rspot ~ rfuture + lresid'
results = smf.ols(formula, data).fit()
print(results.summary())



# In[5]:


# specification 2: rspot rfutures L.rspot L.rfutures
formula = 'rspot ~ rfuture + lspot + lfuture'
results = smf.ols(formula, data).fit()
print(results.summary())


# In[6]:


from statsmodels.tsa.vector_ar import vecm

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'                     'QMF Book/book Ran/data files new/Book4e_data/'
data = pd.read_excel(abspath + 'FRED.xls', index_col=0)



# In[7]:


plt.figure(1, dpi=600)
plt.plot(data['GS3M'], label='GS3M')
plt.plot(data['GS6M'], label='GS6M')
plt.plot(data['GS1'], label='GS1')
plt.plot(data['GS3'], label='GS3')
plt.plot(data['GS5'], label='GS5')
plt.plot(data['GS10'], label='GS10')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()



# In[8]:


import warnings
warnings.filterwarnings('ignore')

# VECM select appropriate lag orders
model = vecm.select_order(data,maxlags=12)
print(model.summary())

# In[9]:


vec_rank1 = vecm.select_coint_rank(data, det_order = 1, k_ar_diff = 1,
                             method = 'trace', signif=0.01)
print(vec_rank1.summary())


# In[10]:


vec_rank2 = vecm.select_coint_rank(data, det_order = 1, k_ar_diff = 1,
                             method = 'maxeig', signif=0.01)
print(vec_rank2.summary())



# In[11]:


# VECM 
model = vecm.VECM(data, k_ar_diff=1,coint_rank=5,deterministic='co')
res = model.fit()
print(res.summary())

