s0=100.
r=0.05
v0=0.1
kappa=3.0
theta=0.25
sigma=0.1
rho=0.6
T=1.0

import numpy as np
corr_mat = np.zeros((2,2))
corr_mat[0,:]=[1.0,rho]
corr_mat[1,:]=[rho,1.0]
cho_mat=np.linalg.cholesky(corr_mat)

cho_mat

M=50
I=10000
dt=T/M

import numpy.random as npr
ran_num=npr.standard_normal((2,M+1,I))

v=np.zeros_like(ran_num[0])
vh=np.zeros_like(v)

v[0]=v0
vh[0]=v0

import math
for t in range(1,M+1):
    ran=np.dot(cho_mat,ran_num[:,t,:])
    vh[t]=(vh[t-1]+kappa*(theta-np.maximum(vh[t-1],0))*dt
           +sigma*np.sqrt(np.maximum(vh[t-1],0))*math.sqrt(dt)*ran[1])

v=np.maximum(vh,0)

S=np.zeros_like(ran_num[0])
S[0]=s0
for t in range(1,M+1):
    ran=np.dot(cho_mat,ran_num[:,t,:])
    S[t]=S[t-1]*np.exp((r-0.5*v[t])*dt+np.sqrt(v[t])*ran[0]*np.sqrt(dt))


import matplotlib.pyplot as plt
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,6))
ax1.hist(S[-1],bins=50)
ax1.set_xlabel('index level')
ax1.set_ylabel('frequency')
ax2.hist(v[-1],bins=50)
ax2.set_xlabel('volatility')

fig,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(10,6))
ax1.plot(S[:,:10],lw=1.5)
ax1.set_ylabel('index level')
ax2.plot(v[:,:10],lw=1.5)
ax2.set_xlabel('time')
ax2.set_ylabel('volatility')
plt.show()