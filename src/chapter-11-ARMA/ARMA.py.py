
# In[1]:


import pandas as pd
import statsmodels.tsa.api as smt
import pickle

abspath = 'C:/Users/tao24/OneDrive - University of Reading/PhD/'    
                 'QMF Book/book Ran/data files new/Book4e_data/'   
data = pd.read_excel(abspath + 'UKHP.xls', index_col=0)
data['dhp'] = data['Average House Price'].transform(lambda x : (x - x.shift(1))/x.shift(1)*100)
data = data.dropna()
data.head()


# In[2]:


with open(abspath + 'UKHP.pickle', 'wb') as handle:
    pickle.dump(data, handle)


# In[3]:


acf,q,pval = smt.acf(data['dhp'],nlags=12,qstat=True)
pacf = smt.pacf(data['dhp'],nlags=12)

correlogram = pd.DataFrame({'acf':acf[1:],
                            'pacf':pacf[1:],
                            'Q':q,
                            'p-val':pval})
correlogram



# In[4]:


res = smt.ARIMA(data['dhp'], order=(1,0,1)).fit()
print(res.summary())



# In[5]:


print(res.aic)
print(res.bic)



# In[6]:


smt.ArmaProcess.from_estimation(res).isinvertible


# In[7]:


smt.ArmaProcess.from_estimation(res).isstationary



# In[8]:


import warnings
warnings.filterwarnings('ignore')

res1 = smt.arma_order_select_ic(data['dhp'],
                                max_ar=5, max_ma=5, ic=['aic', 'bic'],                               fit_kw={'method':'css-mle',
                                'solver':'bfgs'}) 


# In[9]:


print('AIC')
print(res1.aic)
print('SBIC')
print(res1.bic)


# In[10]:


print(res1.aic_min_order)
print(res1.bic_min_order)

