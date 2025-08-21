# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:30:47 2024

@author: dweiss38
"""

import copy
import tqdm
import math
import numpy as np
import itertools
import scipy.io

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial import ConvexHull
from scipy.signal import find_peaks
from scipy.optimize import minimize, Bounds
from skimage.measure import block_reduce
from matplotlib import cm
from scipy.optimize import nnls

from ssm.util import softplus, inv_softplus
from ssmdm.misc import generate_clicks_D

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

Pastel1 = plt.get_cmap('Pastel1')
newcolors = tuple([Pastel1.colors[c] for c in [8,0,1]])
newcmap = ListedColormap(newcolors)

def prepForSSID(trials, fsAI, fs_bin, timerange=None,ignoreNeurons=None):
    Ain = list()
    spks_t = list()
    for trial in trials:
        Ain.append(trial[0])
        spks_t.append(trial[1])
    # Ain = trials[:, 0]
    # spks_t = trials[:, 1]
    if timerange is None:
        start = 0
        stop = int(Ain[0].shape[1]/(fsAI/fs_bin))
    else:
        start = int(timerange[0]*fs_bin)
        stop = int(timerange[1]*fs_bin)
    us = []
    ys = []
    # loop through all trials
    for j in range(len(Ain)):
        u = np.zeros((np.int32(Ain[0].shape[1]/(fsAI/fs_bin)),4))
        y = np.zeros((np.int32(Ain[0].shape[1]/(fsAI/fs_bin)),len(spks_t[0])))
        for num,i in enumerate([0,1,3,4]):
            stims, _ = find_peaks(Ain[j][i,:], height=2, distance=20)
            idx = np.int32(stims/fsAI*fs_bin)
            u[idx,num] = 1
            # Lstims, _ = find_peaks(Ain[j][3,:], height=2, distance=20)
            # RLick, _ = find_peaks(Ain[j][:,4], height=2, distance=20)
            # LLick, _ = find_peaks(Ain[j][:,1], height=2, distance=20)
        
        # loop through all neurons
        for k in range(len(spks_t[j])):
            # Bin spike counts in the jth trial
            if spks_t[j][k] is not None:
                b, _ = np.histogram(spks_t[j][k], bins=int(Ain[0].shape[1] / (fsAI/fs_bin)), range=(0,6.5))
                y[:,k] = b
            
        u = u[start:stop,:]
        y = y[start:stop,:]
        if ignoreNeurons is not None:
            y = np.delete(y,ignoreNeurons,axis=1)
        
        
        us.append(u)
        ys.append(y)
    
    return us, ys

def load_trial(data,expnum,fs_bin=100,timerange=[1.95,4],ignoreMiss=True,nTrain=None,ignoreEarly=None,includeNeurons=None,m73=False):
    general = scipy.io.loadmat('Y:\stanley\Data\David\ExperimentData\DW011_4\Ephys\Processed\_GeneralInfo.mat')["general"]
    if not m73:
        tri = data["sessions"][0,expnum]
        trials = [[tr[0], list(tr[1][0,:])] for tr in tri ]
        behavior = data["behaviorResults"][0,expnum][0,0]
        fsAI = general["fsAI"][0][0][0][0]
    else:
        trials = data["sessions"][expnum]
        behavior = data["behaviorResults"][expnum]
        fsAI = general["fsAI"][0][0][0][0]
    
    allneurons = list(range(1,(len(trials[0][1]))+1))
    if not includeNeurons:
        includeNeurons = list(range(1,(len(trials[0][1]))+1))#[3,5,8,7,15,25,28,27,29,31,9]#

    ignoreNeurons = np.where(np.isin(allneurons,includeNeurons,invert=True))[0]
    #ignoreNeurons = [2,7,10,18]
    inputs, observations = prepForSSID(trials, fsAI, fs_bin, timerange=timerange,ignoreNeurons=ignoreNeurons)
    
    smoothstd = 45
    
    ys = [np.int32(y) for y in observations]
    ys_train = ys[:nTrain]
        
    us = [inp[:,[0,2]] for inp in inputs]
    us_train = us[:nTrain]
    
    bs = [inp[:,[1,3]] for inp in inputs]
    bs_train = bs[:nTrain]
    
    if ignoreMiss:  
        if ignoreEarly:
            indexes = np.where(np.logical_and(np.squeeze(behavior['firstLick'])>(2+ignoreEarly)*fsAI,np.squeeze(behavior['Miss']) == 0))[1]
        else:
            indexes = np.where(np.squeeze(behavior['Miss']) == 0)[0]
        ir = np.squeeze(behavior['IR'])[indexes]
        il = np.squeeze(behavior['IL'])[indexes]
        diff = np.squeeze(behavior['difference'])[indexes]
        cr = np.squeeze(behavior['CR'])[indexes]
        cl = np.squeeze(behavior['CL'])[indexes]
        leftresponse = np.squeeze(behavior['LeftResponse'])[indexes]
        rightresponse = np.squeeze(behavior['RightResponse'])[indexes]
        firstlick = np.squeeze(behavior['firstLick'])[indexes]
        abort = np.squeeze(behavior['Aborted'])[indexes]
        bs = [bs[idx] for idx in indexes]
        bs_train = [bs_train[idx] for idx in indexes]
        
        ys = [ys[idx] for idx in indexes]
        ys_train = [ys_train[idx] for idx in indexes]
        
        us = [us[idx] for idx in indexes]
        us_train = [us_train[idx] for idx in indexes]
    
    if not ignoreMiss:
        RightResponse = np.squeeze(behavior['RightResponse'])
        LeftResponse = np.squeeze(behavior['LeftResponse'])
        
        difference = np.squeeze(behavior['difference'])
        firstLick = np.squeeze(behavior['firstLick'])
        aborted = np.squeeze(behavior['Aborted'])
       
        signedStrength = difference/(np.max(np.abs([np.max(difference), np.min(difference)])))
        
        # Right == 2, Left == 1
        choice = 2*np.int32(RightResponse) + 1*np.int32(LeftResponse)
        
        b,a = np.histogram(signedStrength,np.linspace(-1,1,8))
        
        # Construct mean responses
        # Population average Responses
        distances = np.abs(np.tile(signedStrength, (len(a), 1)).T - a)
        cond = np.argmin(distances, axis=1)
        IncorrectRight = np.squeeze(behavior['IR'])
        IncorrectLeft = np.squeeze(behavior['IL'])
        CorrectRight = np.squeeze(behavior['CR'])
        CorrectLeft = np.squeeze(behavior['CL'])
        Miss = np.squeeze(behavior['Miss'])
    else:
        RightResponse = rightresponse[None,:]
        LeftResponse = leftresponse[None,:]
        difference = diff
        firstLick = firstlick[None,:]
        aborted = abort[None,:]
        signedStrength = difference/(np.max(np.abs([np.max(difference), np.min(difference)])))
        
        # Right == 2, Left == 1
        choice = 2*np.int32(RightResponse) + 1*np.int32(LeftResponse)
        
        b,a = np.histogram(signedStrength,np.linspace(-1,1,8))
        
        # Construct mean responses
        # Population average Responses
        distances = np.abs(np.tile(signedStrength, (len(a), 1)).T - a)
        cond = np.argmin(distances, axis=1)
        IncorrectRight = ir
        IncorrectLeft = il
        CorrectRight = cr
        CorrectLeft = cl
        Miss = np.zeros((1,len(CorrectLeft)))
            
    behaviorResults = {'RightResponse': RightResponse,
                       'LeftResponse': LeftResponse,
                       'difference': difference,
                       'firstLick': firstLick,
                       'signedStrength': signedStrength,
                       'cond': cond,
                       'choice': choice,
                       'IncorrectRight':IncorrectRight,
                       'IncorrectLeft':IncorrectLeft,
                       'CorrectRight': CorrectRight,
                       'CorrectLeft' : CorrectLeft,
                       'Aborted': aborted,
                       'Miss': Miss}
    
    return us,ys,bs,us_train,ys_train,bs_train,behaviorResults

def plot_Ain(trials,idx):
    Ain = trials[:, 0]
    plt.plot(Ain[idx].T+np.array([0, 5, 10, 15, 20, 25]))
    plt.legend(['LGalv','LLick','LReward','RGalv','RLick','RReward'])
    
def prepForTrialRasters(trials, fsAI, fs_bin, timerange=None, ignoreNeurons=None):
    
    Ain = trials[:, 0]
    spks_t = trials[:, 1]
    if timerange is None:
        start = 0
        stop = int(Ain[0].shape[1]/(fsAI/fs_bin))
    else:
        start = int(timerange[0]*fs_bin)
        stop = int(timerange[1]*fs_bin)

    ys = []
    # loop through all neurons
    for k in range(spks_t[0].shape[1]):
        if np.isin(k,ignoreNeurons):
            continue
        y = np.zeros((np.int32(Ain[0].shape[1]/(fsAI/fs_bin)),len(Ain)))
        # loop through all trials
        for j in range(len(Ain)):     

            # Bin spike counts in the jth trial
            b, _ = np.histogram(spks_t[j][0,k], bins=int(Ain[0].shape[1] / (fsAI/fs_bin)), range=(0,6.5))
            y[:,j] = b
            
        y = y[start:stop,:]

        ys.append(y)
    
    return ys

def split_lr_choice(ys, behavior):
    RightResponse = behavior[0,0]['RightResponse'][0]
    LeftResponse = behavior[0,0]['LeftResponse'][0]
    ys_L = []
    ys_R = []
    for y in ys:
        ys_L.append(y[:,np.where(LeftResponse)[0]])
        ys_R.append(y[:,np.where(RightResponse)[0]])
        
    return ys_L, ys_R
    
def plotRaster(ys,index,ax):
    sns.despine()
    for n in range(ys[index].shape[1]):
        ax.eventplot(np.where(ys[index][:,n]>0)[0], linelengths=0.5, lineoffsets=1+n,color='k')
        #.scatter(np.where(ys[index][:,n]>0)[0],np.ones_like(np.where(ys[index][:,n]>0)[0])+n,color='k',marker='|')
    sns.despine()
    #ax.set_yticks()
    ax.set_title("spikes %d" % index)

def make_exp_kern(width, dt):
    time = np.linspace(0,4*width,int(4*width/dt))
    return np.exp(-time/(width))

def make_halfgauss_kern(sigma, dt):
    time = np.linspace(-4*sigma,4*sigma,int((8*sigma)/dt)+1)
    gauss = 1/(np.sqrt(2*math.pi)*sigma)*np.exp(-time**2/(2*sigma**2))*dt
    return gauss[int((gauss.shape[0]-1)/2):]*2

def make_gauss_kern(sigma, dt):
    time = np.linspace(-4*sigma,4*sigma,int((8*sigma)/dt)+1)
    gauss = 1/(np.sqrt(2*math.pi)*sigma)*np.exp(-time**2/(2*sigma**2))*dt
    return gauss
    
def smooth(x, kernel,mode='same',axis=0):
    kernel = kernel[:x.shape[0]]
    return np.apply_along_axis(np.convolve, axis, x, kernel, mode=mode)

def moveavg(x, window_sz,mode='same'):
    window = np.ones(window_sz)
    window = window[:x.shape[0]]
    window /= window_sz
    return np.apply_along_axis(np.convolve, 0, x, window, mode=mode)


def plot_labeled_trajectories(X,labels,num_to_plot=None,alpha=0.3):
    idxr = np.where(labels == 1)[0]
    idxl = np.where(labels == 2)[0]
    idxm = np.where(labels == 0)[0]
    rmean = np.mean(X[labels==1,:],axis=0)
    lmean = np.mean(X[labels==2,:],axis=0)
    mmean = np.mean(X[labels==0,:],axis=0)
    if num_to_plot is not None:
        idxr = idxr[np.random.randint(0,len(idxr),size=num_to_plot)]
        idxl = idxl[np.random.randint(0,len(idxl),size=num_to_plot)]
        idxm = idxm[np.random.randint(0,len(idxm),size=num_to_plot)]
    plt.figure()
    plt.plot(X[idxr,:].T,color='r',alpha=alpha)
    plt.plot(X[idxl,:].T,color='b',alpha=alpha)
    plt.plot(X[idxm,:].T,color='g',alpha=alpha)
    plt.plot(rmean,'r',label='Right Lick')
    plt.plot(lmean,'b',label='Left Lick')
    plt.plot(mmean,'g',label='Miss')

    
# def plot_trajectory(q_lem,model,us,ys,bs,fs,tr=0,legend=False):
    
#     q_x = q_lem.mean_continuous_states[tr]
#     zhat = model.most_likely_states(q_x, ys[tr], input=us[tr])
#     yhat = model.smooth(q_x, ys[tr], input=us[tr])
#     zhat = model.most_likely_states(q_x, ys[tr], input=us[tr])
#     t = np.linspace(0,q_x.shape[0]/fs,q_x.shape[0])
    
def make_feature_matrix(q_x,nBins):
    
    X = np.empty([0,nBins*q_x.mean_continuous_states[0].shape[1]])
    for trialnum in range(len(q_x.mean_continuous_states)):
        xhat = q_x.mean_continuous_states[trialnum]
        xhat_reduced = block_reduce(xhat,(np.int64(len(xhat)/nBins),1),np.mean)
        X = np.concatenate([X,xhat_reduced.T.flatten()[np.newaxis]],axis=0)
        #yhat = rslds.most_likely_states(xhat, d[trialnum])
    return X

def plot_est_firing_rates(q_lem,model,ys,us=None,tr=0):
    q_x = q_lem.mean_continuous_states[tr]
    #rates = softplus(model.emissions.Cs[0] @ q_x.T+model.emissions.ds.T).T *0.01
    #plt.plot(rates)
    if us is not None:
        yhat = model.smooth(q_x, ys[tr], input=us[tr])
    else:
        yhat = model.smooth(q_x, ys[tr])
    plt.plot(yhat)
    

def plot_most_likely_dynamics(model,
    xlim=(-4, 4), ylim=(-3, 3), nxpts=30, nypts=30,
    alpha=0.8, ax=None, figsize=(3, 3)):
    
    color_names = ["windows blue", "red", "amber", "faded green"]
    colors = sns.xkcd_palette(color_names)
    K = model.K
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    log_Ps = model.transitions.log_transition_matrices(
        xy, np.zeros((nxpts * nypts, 0)), np.ones_like(xy, dtype=bool), None)
    z = np.argmax(log_Ps[:, 0, :], axis=-1)
    z = np.concatenate([[z[0]], z])

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax

def normalized_error(param_id, param_true):
    ## Ahmadipour et al 2024
    return np.linalg.norm(param_id - param_true)/np.linalg.norm(param_true)

def total_normalized_error(model, plant, D=None, M=None):
    if D is None:
        D = model.D
    if M is None:
        M = model.M
    if plant.D != D:
        As = []
        Vs = []
        for k in range(plant.K):
            As.append(plant.dynamics.As[k][:D,:D])
            Vs.append(plant.dynamics.Vs[k][:D,:M])
        As_error = normalized_error(model.dynamics.As[:,:D,:D], As)
        Vs_error = normalized_error(model.dynamics.Vs[:,:D,:M], Vs)
        Cs_error = normalized_error(model.emissions.Cs[:,:,:D], plant.emissions.Cs[:,:,:D])
        ds_error = normalized_error(model.emissions.ds, plant.emissions.ds)
    else:
        As_error = normalized_error(model.dynamics.As, plant.dynamics.As)
        Vs_error = normalized_error(model.dynamics.Vs, plant.dynamics.Vs)
        Cs_error = normalized_error(model.emissions.Cs, plant.emissions.Cs)
        ds_error = normalized_error(model.emissions.ds, plant.emissions.ds)
    return As_error,Vs_error,Cs_error,ds_error

def rmse(x, xhat):
    return np.sqrt(np.mean((x-xhat)**2))

def PoissonNLL(rate,spikes,eps=1e-6):
    return np.sum(rate - spikes*np.log(rate))

# def normalized_mode_error(A_id,A_true):

def find_similarity_transform(system, model, optimize=False):
    # Sample state trajectories for simulated and model systems
    # Find the similarity transformation that minimizes the difference
    # between sampled trajectories following Sani, ..., Shanechi 2021
    
    T = 150 # number of time bins
    trial_time = 1.5 # trial length in seconds
    dt = 0.01 # bin size in seconds
    N_samples = 80
    
    # input statistics
    total_rate = 40 # the sum of the right and left poisson process rates is 40
    xs = np.empty((0,2))
    xhats = np.empty((0,2))
    for smpl in range(N_samples):

        # randomly draw right and left rates
        rate_r = np.random.randint(0,total_rate+1)
        rate_l = total_rate - rate_r
        rates = [rate_r,rate_l]

        # generate binned right and left clicks
        u = generate_clicks_D(rates,T=trial_time,dt=dt)

        # input is sum of u_r and u_l
        u = 1.0*np.array(u).T
        z, x, y = system.sample(T, input=u)
        zhat,xhat,yhat = model.sample(T,input=u)
        
        xs = np.concatenate([xs,x],axis=0)
        xhats = np.concatenate([xhats,xhat],axis=0)
    
    if optimize:
        def _obj(T_flat): return np.sum(np.linalg.norm(np.reshape(T_flat,(xhats.shape[1],xhats.shape[1])) @ xhats.T - xs.T)**2)
        x0 = (np.random.uniform(size=(2,))*np.eye(2)).flatten()+1e-6
        res = minimize(_obj,x0)
        T = np.reshape(res.x,(xhats.shape[1],xhats.shape[1]))
    
    T = np.linalg.inv(xhats.T @ xhats) @ xhats.T @ xs
    
    return T

def transform_system(model, T):
    Tinv = np.linalg.inv(T)
    for k, (A, B, C) in enumerate(zip(model.dynamics.As, model.dynamics.Vs, 
                                      model.emissions.Cs)):
        model.dynamics.As[k] = T @ A @ Tinv
        model.dynamics.Vs[k] = T @ B
        model.emissions.Cs[k] = C @ Tinv
        
def plot_with_errors(x,y,std,xlabel=None,ylabel=None,title=None):
    if len(y.shape) < 2:
        y = y[np.newaxis].T
        std = std[np.newaxis].T
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for jj in range(y.shape[1]):
        plt.plot(x,y[:,jj],color=colors[jj],alpha=0.9)
        plt.fill_between(x,y[:,jj]-std[:,jj], y[:,jj]+std[:,jj], facecolor=colors[jj], alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

import numpy.random as rng
def generate_white_noise(mean, sigma, size):
    return mean + sigma * rng.standard_normal(size=size)



def get_u0(x0, model, Mphys, Mopto, bounds=None):
    if bounds is None:
        u0,_,_,_ = np.linalg.lstsq(model.dynamics.Vs[0][:,Mphys:],x0[1:].T - model.dynamics.As[0] @ x0[:-1].T)
    else:
        #print('Contrained optimization for u*...')
        A = model.dynamics.As[0]
        B = model.dynamics.Vs[0][:,Mphys:]
        Nt = len(x0)-1
        Nu = Mopto
        uinit = np.zeros((Mopto,Nt)).flatten()
        def _obj(ut):
            ut = np.reshape(ut,(Nu,Nt))
            return np.linalg.norm(B @ ut + A @ x0[:-1].T - x0[1:].T)**2
       
        def jac(ut):
            ut = np.reshape(ut,(Nu,Nt))
            return (2*B.T @ (B @ ut + A @ x0[:-1].T - x0[1:].T)).flatten()

        def hess(ut):
            ut = np.reshape(ut,(Nu,Nt))
            return (2*B.T @ (B @ ut)).flatten()
        boundsoptim = Bounds([bounds[0]] * Nt*Nu, [bounds[1]] * Nu*Nt)
        res = minimize(_obj, uinit, method='trust-constr', 
                       jac=jac, hess=hess, bounds=boundsoptim)
        u0 = np.reshape(res.x,(Mopto,Nt))
    return np.vstack((np.zeros((1,Mopto)),u0.T))
    
def make_x0(x_d,D,target=0):
    x0 = np.zeros((len(x_d),D))
    x0[:,target] = x_d
    return x0

def make_xd(N,T,xf=1):
    x_d = np.ones((T,))
    x_d[0:N] = np.linspace(0,1,N)
    x_d *= xf
    return x_d


def nParams(model):
    k = 0
    for params in model.params:
        for param in params:
            k += param.size
    
    return k

def BIC(k,n,NLL):
    return k*np.log(n) + 2*NLL

# #%% Effect of input on dynamics
# beta = np.abs(np.random.randn(25,2)) * np.array([1,-1])
# C = latent_acc.emissions.Cs[0]
# delta = np.abs(latent_acc.emissions.ds[0])
# u = np.vstack((np.linspace(0.01,5),np.linspace(0.01,5)))
# x = np.linalg.inv(C.T@C)@C.T @ (np.log(np.exp(beta@u+delta[:,None])-1)-latent_acc.emissions.ds[0][:,None])

# x = np.linalg.inv(C.T@C)@C.T @ ((beta@u+delta[:,None])-latent_acc.emissions.ds[0][:,None])

# B = np.linalg.inv(C.T@C)@C.T @ beta
    
