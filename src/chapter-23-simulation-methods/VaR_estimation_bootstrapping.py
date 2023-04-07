
# In[1]:


import pandas as pd
import numpy as np
from arch import arch_model
from numpy import sqrt, exp, std, mean
import statsmodels.api as sms

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'     
                'QMF Book/book Ran/data files new/Book4e_data/'
data = pd.read_excel(abspath + 'sp500_daily.xlsx',index_col=[0])

def LogDiff(x):
    x_diff = np.log(x/x.shift(1))
    x_diff = x_diff.dropna()
    return x_diff

data['rf'] = LogDiff(data['Close'])
data = data.rename(columns={'Close':'sp500'})
data = data.dropna()



# In[2]:


res = arch_model(data['rf'], vol='GARCH').fit()

mu = res.params['mu']
resi = res.resid

hsq = res.conditional_volatility # square root of conditional variance
sres = resi/hsq



# In[3]:


def bootstrap_resample(X, n=None):
    """ 
    Bootstrap resample an array_like
    
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample



# In[4]:


minimum = np.zeros(10000)
maximum = np.zeros(10000)
for n in range(10000):
    sres_b = bootstrap_resample(sres)
       
    # calculate multiple-step-ahead forecast by built-in function
    forecasts = res.forecast(horizon=10)
    forecasts_res = forecasts.variance.dropna().T
    
    rtf = mu + sqrt(forecasts_res).values*sres_b[:10].to_frame().values
    
    sp500_f = np.zeros(10)
    for i in range(10):
        if i == 0:
            sp500_f[i] = data['sp500'][-1]*exp(rtf[i])
        else:
            sp500_f[i] = sp500_f[i-1]*exp(rtf[i])

    minimum[n] = min(sp500_f)
    maximum[n] = max(sp500_f)



# In[5]:


long = np.log(minimum/2815.62)
short = np.log(maximum/2815.62)

print('mcrrl:{}'.format(1-exp(-1.645*std(long)+mean(long))))
print('mcrrs:{}'.format(exp(1.645*std(short)+mean(short))-1))


