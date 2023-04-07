

# In[1]:


import scipy as sp
from numpy import average, exp

# parameters input
S0 = 6289.70
K = 6500 # excercise price
T = 0.5 # maturity in years
r = 0.0624 # risk-free rate
d = 0.0242 # dividend yield
sigma = 0.2652 # implied volatility
n_simulation = 50000 # number of simulations
n_steps = 125
dt = T/n_steps
call = sp.zeros([n_simulation],dtype=float)
put = sp.zeros([n_simulation],dtype=float)

for j in range(0, n_simulation):
    ST=S0
    total=0
    
    for i in range(0,int(n_steps)):
        e=sp.random.normal()
        ST=ST*sp.exp((r-d-0.5*sigma**2)*dt+sigma*sp.sqrt(dt)*e)
        total+=ST
    
    price_average=total/n_steps
    call[j]=max(price_average-K,0)*exp(-r*T)
    put[j]=max(K-price_average,0)*exp(-r*T)

call_price=average(call)
put_price=average(put)

print('call price = ', round(call_price,2) )
print('put price = ', round(put_price,2) )


 
