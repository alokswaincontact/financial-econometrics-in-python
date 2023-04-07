

# In[1]:


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'   
                  'QMF Book/book Ran/data files new/Book4e_data/'
data = pd.read_excel(abspath + 'UKHP.xls',index_col=[0])

data['dhp'] = data['Average House Price'].transform(lambda x : (x - x.shift(1))/x.shift(1)*100)

data = data.dropna() 


# In[2]:


model = sm.tsa.MarkovRegression(data['dhp'], k_regimes=2,
                                switching_variance=True)
res = model.fit()
print(res.summary())


# In[3]:


print(res.expected_durations)


# In[4]:


plt.figure(1,dpi = 600)
plt.plot(res.smoothed_marginal_probabilities[1])
plt.title('Smoothed State probabilities')
plt.show()

