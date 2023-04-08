
# In[1]:


import pickle
from linearmodels.system import IVSystemGMM
from linearmodels.iv import IVGMM
import statsmodels.api as sm
import pandas as pd

abspath = '../../data/'

with open(abspath + 'macro.pickle', 'rb') as handle:
    data = pickle.load(handle)

# In[2]:


# GMM, specification 1
formula = 'inflation ~ 1 + dprod + dcredit + dmoney + [rsandp ~ rterm + dspread]'
mod = IVGMM.from_formula(formula, data, weight_type='unadjusted')
res1 = mod.fit(cov_type='robust')
print(res1.summary)

# In[3]:


# GMM, specification 2
formula = 'rsandp ~ 1 + dprod + dcredit + dmoney + [inflation ~ rterm + dspread]'
mod = IVGMM.from_formula(formula, data, weight_type='unadjusted')
res2 = mod.fit(cov_type='robust')
print(res2.summary)

