
# In[1]:


import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.tsatools import add_trend 



# In[2]:


# Simulate an AR(1) process with alpha = 1
def simulate_AR(obs,alpha,w,const=False,trend=False):
    '''
    Simulate an AR(1) process.

    Parameters:
    -----------
    obs : the required number of observations.

    alpha: the coefficient of y.
    
    w: a set of error terms that follow a normal distribution.
       
    const: the boolean, whether to add a constant term in the formula.
           default is false.
    
    trend: the boolean, whether to add a linear trend in the formula.
           default is false.
    
    Returns:
    --------
    res.tvalues: the t-statistic of the regression instance under each
                 simulation. 
    '''

    obs = int(obs)
    a = int(alpha)
    y = np.zeros(shape=(obs))
    
    for t in range(obs):
        if t == 0:
            y[t] = 0
        else:
            y[t] = a*y[t-1] + w[t]
    
    dy = np.diff(y)
    y = y[:-1]
    
    if (const is False) & (trend is False):
        res = sm.OLS(dy,y).fit()
        return res.tvalues
    
    if (const is True) & (trend is False):
        y = add_trend(y, trend='c')
        res = sm.OLS(dy,y).fit()
        return res.tvalues[0]

    if (const is True) & (trend is True):
        y = add_trend(y, trend='ct')
        res = sm.OLS(dy,y).fit()
        return res.tvalues[0]



# In[3]:


## Model/Experiment ------------------------------

t1 = np.zeros(shape=(50000))
t2 = np.zeros(shape=(50000))
t3 = np.zeros(shape=(50000))
for i in range(50000):
    errors = np.random.normal(size=1000) 
    t1[i] = simulate_AR(obs=1000, alpha=1, w=errors, const=False, trend=False)
    t2[i] = simulate_AR(obs=1000, alpha=1, w=errors, const=True, trend=False)
    t3[i] = simulate_AR(obs=1000, alpha=1, w=errors, const=True, trend=True)

print('No const or trend: 1% {}'.format(np.percentile(t1,1)) )
print('No const or trend: 5% {}'.format(np.percentile(t1,5)) )
print('No const or trend: 10% {}'.format(np.percentile(t1,10)) )

print('Const but no trend: 1% {}'.format(np.percentile(t2,1)) )
print('Const but no trend: 5% {}'.format(np.percentile(t2,5)) )
print('Const but no trend: 10% {}'.format(np.percentile(t2,10)) )

print('Const and trend: 1% {}'.format(np.percentile(t3,1)) )
print('Const and trend: 5% {}'.format(np.percentile(t3,5)) )
print('Const and trend: 10% {}'.format(np.percentile(t3,10)) )


