
# In[1]:


import pickle
from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

abspath = '../../data/'
with open(abspath + 'currencies.pickle', 'rb') as handle:
    data = pickle.load(handle)

# Sampling    
data_in_the_sample = data.loc[:'2016-08-02','rjpy']     
data_out_of_the_sample = data.loc['2016-08-03':,'rjpy']



# In[2]:


am = arch_model(data['rjpy'], vol='GARCH') 

cvar_rjpy_stat = {}
for date in data_out_of_the_sample.index:
    res = am.fit(last_obs = date, disp='off')
    forecasts = res.forecast(horizon=1)
    forecasts_res = forecasts.variance.dropna()
    cvar_rjpy_stat[date] = forecasts_res.iloc[1]

cvar_rjpy_stat = pd.DataFrame(cvar_rjpy_stat).T

# In[3]:


res = am.fit(last_obs = '2016-08-03', disp='off')
forecasts = res.forecast(horizon=len(data_out_of_the_sample))
forecasts_res = forecasts.variance.dropna()

cvar_rjpy_dyn = pd.DataFrame(data = forecasts_res.iloc[1].values,                             columns=['dynamic forecasting'],                             index=data_out_of_the_sample.index)


# In[4]:


plt.figure(1, dpi = 600)
plt.plot(cvar_rjpy_stat, label='static forecast')
plt.plot(cvar_rjpy_dyn, label='dynamic forecast')

plt.xlabel('Date')
plt.ylabel('Forecasts')
plt.legend()
plt.show()
