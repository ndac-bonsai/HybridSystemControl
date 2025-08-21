# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:42:40 2025

@author: dweiss38
"""

from main1 import OnlineSLDSModel
import numpy as np
import copy
import matplotlib.pyplot as plt

D=2
M=2
plant = OnlineSLDSModel(N=10,D=2,K=2,M=2,transitions="recurrent_only",dynamics="gaussian",emissions="poisson",dt=0.01,alpha_variance=0.99,x_variance=1.1,SigmasScaler=0.1,with_noise=True)
#model.dynamics.bs = np.zeros((1,D))
model = copy.deepcopy(plant)
y=np.array([1,4,2,0,1])
print(model.x)
# model.sample(u)
# print(model.z)
# print(model.x)
# print(model.y)

# model.sample(u)

# #estimator = OnlineHMMKFEstimator(N=5,D=2,K=3,M=1,As=[0.280012667053292, 0.749395026864805, -0.749395026864805, 0.280012667053292, 0.41157561343426, 0.686006934677932, -0.686006934677932, 0.41157561343426, 0.213332738065125, 0.771031220424722, -0.771031220424722, 0.213332738065125],Bs=[0.135971856173445, -0.015482582044147, 0.535952442684655, 1.14148681421371, 0.410875082706607, 1.28480443642902],Cs=[-0.262919266490783, -0.617867550128008, -1.93768502707253, -0.749266526390563, 1.56299323715796, 1.07747269280039, -1.095936230246, -1.26117065952602, -0.564981833063247, 1.35037388321917],Fs=[-0.94730122436999, 0.757828530092683, -0.0201173513183226, 1.04596328055353, -1.05958378527577],ds=[-1.88345055416675, -1.78786446838797, -0.859717460971858, 1.15370832199875, 0.978482996478272],sigmasq_init=[1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],Qs=[1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],Rs=[0, 0, 200, 0, 0, 200],r=[0, -200, -200],pi0=[0.333333333333333, 0.333333333333333, 0.333333333333333],alpha_variance=0.001,x_variance=[1, 0],dt=0.01,constant=False)

# estimator.filt(u,y)
# estimator.filt(u,y)
# estimator.filt(u,y)
# estimator.filt(u,y)
# estimator.filt(u,y)
# #z,x,y = sampleModel(model,u)
t_sim = 1
u = np.vstack((np.zeros((50,M)), np.ones((1,M))*np.sin(2*np.pi*np.linspace(0,t_sim,100*t_sim))[:,None]))[:,:M]

xs = []
xs.append(plant.x)
xhats = []
for inp in u:
    inputs = np.array([inp])
    plant.sample(inputs)
    xs.append(plant.x)
    model.filt(inputs,plant.y)
    ustar = model.controller.ustar(model.xhat[-1],model.zhat,0)
    xhats.append(model.xhat[-1])
    print(ustar)
    
plt.figure()
plt.plot(xs)
plt.figure()
plt.plot(xhats)


#%% test NDAC
D=2
plant = OnlineSLDSModel(N=10,D=2,K=2,M=2,transitions="recurrent_only",dynamics="gaussian",emissions="poisson",dt=0.01,alpha_variance=0.99,x_variance=1.1,SigmasScaler=0.1,with_noise=True)
#model.dynamics.bs = np.zeros((1,D))
model = copy.deepcopy(plant)
u = np.array([1,2])
for i in range(100):
    inputs = np.squeeze(u)
    plant.sample(inputs)
    xs.append(plant.x)
    model.filt(inputs,plant.y)
    ustar = model.controller.ustar(model.xhat[-1],model.zhat,0)
    xhats.append(model.xhat[-1])
    u=ustar[None,:]
    print(ustar)