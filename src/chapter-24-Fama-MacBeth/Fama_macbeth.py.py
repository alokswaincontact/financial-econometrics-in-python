
# In[1]:


import pandas as pd
import statsmodels.api as sm
import numpy as np

abspath = '../../data/'
data1 = pd.read_csv(abspath + 'monthlyfactors.csv',index_col=[0])
data2 = pd.read_csv(abspath + 'vw_sizebm_25groups.csv',index_col=[0])

# subsample
data1 = data1['1980m10':'2010m12']*100
data2 = data2['1980m10':'2010m12']*100

# excess return
for each in data2.columns:
    data2[each] = data2[each]-data1['rf']



# In[2]:


# first stage
x = data1[['smb','hml','umd','rmrf']]
x = sm.add_constant(x)
betas = []
for each in data2.columns:
    y = data2[[each]]
    res = sm.OLS(y,x).fit().params
    res = res.rename(each)
    betas.append(res)
    
betas = pd.concat(betas, axis=1)



# In[3]:


# second stage
x = betas.transpose()[['smb','hml','umd','rmrf']]
x = sm.add_constant(x)
lambdas = []
for each in data2.transpose().columns:
    y = data2.transpose()[[each]]
    res = sm.OLS(y,x).fit().params
    res = res.rename(each)
    lambdas.append(res)
    
lambdas = pd.concat(lambdas, axis=1)



# In[4]:


# final stage
lambdas1 = lambdas.mean(axis=1)
lambdas2 = lambdas.std(axis=1)
lambdas3 = np.sqrt(lambdas.shape[1])*lambdas1/lambdas2



# In[5]:


output = pd.concat([lambdas1,lambdas3],join='inner', axis=1)
output = output.rename(columns={0:'Estimates',
                                1:'t-ratio'})
output

