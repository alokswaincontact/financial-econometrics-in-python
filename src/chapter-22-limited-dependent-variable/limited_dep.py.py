
# In[1]:


import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

abspath = '../../data/'
data = pd.read_excel(abspath + 'MSc_fail.xls')

data = data.rename(columns={'Work Experience':'WorkExperience',
                            'PG Degree':'PGDegree'})
data.head()


# In[2]:


data.info()



# In[3]:


# linear regression
formula = 'Fail ~ Age + English + Female + WorkExperience + Agrade + BelowBGrade + PGDegree + Year2004 + Year2005 + Year2006                 + Year2007'
mod = smf.ols(formula, data)
res = mod.fit()
print(res.summary())



# In[4]:


# logit regression
mod = smf.logit(formula, data)
res = mod.fit(cov_type='HC1') # white robustness
print(res.summary())



# In[5]:


# probit regression
mod = smf.probit(formula, data)
res = mod.fit(cov_type='HC1')
print(res.summary())



# In[6]:


plt.figure(1,dpi = 600)
plt.plot(res.predict())
plt.ylabel('Pr(Fail)')
plt.xlabel('Seqnum')
plt.show()


# In[7]:


print(res.get_margeff().summary())



# In[8]:


mod = smf.logit(formula, data)
res = mod.fit(cov_type='HC1')
print(res.get_margeff().summary())


