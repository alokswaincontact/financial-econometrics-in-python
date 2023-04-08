
# In[1]:


from statsmodels.stats.outliers_influence import reset_ramsey
import statsmodels.formula.api as smf
import pickle

abspath = '../../data/'
with open(abspath + 'macro.pickle', 'rb') as handle:
    data = pickle.load(handle)

data = data.dropna() # drop the missing values for some columns


# In[2]:


formula = 'ermsoft ~ ersandp + dprod + dcredit + dinflation + dmoney + dspread + rterm'
results = smf.ols(formula, data).fit()

reset_ramsey(results,degree=4)

