# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:33:13 2025

@author: dweiss38
"""

import ssm.stats as stats

from ssm.util import logistic, logit, softplus, inv_softplus
from modeling_utils import smooth, make_exp_kern, make_halfgauss_kern, moveavg
from autograd.scipy.special import logsumexp, gammaln
import numpy as np
import copy
import numpy.random as npr
from scipy.linalg import block_diag
import time

class HMMKFEstimator(object):
    def __init__(self, As, Bs, Cs, Ds, ds, sigmasq_init, Qs, Rs, r, pi0, 
                 alpha_variance=0.01,x_variance=100,bin_size=0.01,constant=False):
        self.D = As.shape[1]
        self.M = Bs.shape[2]
        self.N = Cs.shape[1]
        self.K = As.shape[0]
        self.As = As
        self.Bs = Bs
        # Rescaling Cs, ds to account for different observation model
        self.Cs = Cs#np.array([np.log(np.abs(c)) * np.sign(c) for c in Cs]) # Cs #
        self.Ds = Ds #np.array([np.log(np.abs(d)) * np.sign(d) for d in Ds]) # 
        self.ds = ds#np.array([np.log(np.abs(d)) * np.sign(d) for d in ds]) # ds #
        self.sigmasq_init = sigmasq_init
        self.Qs = Qs
        self.Rs = Rs
        self.r = r
        self.pi0 = pi0
        self.constant = constant
        #self.initial_variance = initial_variance
        self.x_variance = x_variance
        self.alpha_variance = alpha_variance
        if type(x_variance) == float:
            self.initial_variance = np.hstack((x_variance*np.ones((1,self.D)), \
                                               alpha_variance*np.ones((1,self.N))))
        else:
            self.initial_variance = np.hstack((x_variance[None,:], \
                                               alpha_variance*np.ones((1,self.N))))  
        self.bin_size = bin_size
        
        self.Qs_inv = [np.linalg.inv(Q) for Q in Qs]
        self.As_inv = [np.linalg.inv(A) for A in As]
        
        self.times = []

        
    def initialize(self):
        self.Ez = self.pi0
        self.J_filt = np.zeros((self.D,self.D))
        self.h_filt = np.zeros(self.D)
        J_ini = np.sum(self.pi0[:,None,None] * np.linalg.inv(self.Qs), axis=0)
        self.J_pred = copy.deepcopy(J_ini)
        self.h_pred = J_ini @ np.zeros(self.D) # Assume zero initial conditions, = h_ini
        self.zhat = 0
        
        self.J_obs = 1 / self.x_variance * np.eye(self.D)
        
        self.log_Z_ini = 0
        self.log_Z_dyn = 0
        self.log_Z_obs = 0
        
        #self.x = self.J_pred @ self.h_pred
        self.x = 0.05*npr.randn(1,self.D) # inititalize x near 0
        self.x_pred = np.zeros_like(self.x)
        
        #self.P = np.sum(pi0[:,None,None] * self.Qs, axis=0)[None,:,:] # Initialize covariance matrix with model params
        if self.constant is False or self.constant == 'softplus' or self.constant == 'softplus1':
            self.P = np.eye(self.D)[None,:,:]
            self.Q = self.initial_variance[0,:self.D] * np.eye(self.D) # initialize update covariance
        else:
            self.P = np.eye(self.D+self.N)[None,:,:]
            self.Q = self.initial_variance * np.eye(self.D + self.N) # initialize update covariance
        self.P_pred = np.zeros_like(self.P)
        self.u = np.zeros(self.M)
        self.y = np.zeros(self.N,dtype=int)
        self.zhat_arr = [0]
        self.alphas = 0.9 * np.ones(self.N)
        self.times = []
    
    
    ### Transitions/HMM things
    def log_transition_matrices(self):
        #log_Ps = np.dot(input[1:], self.Ws.T)[:, None, :]              # inputs
        log_Ps =  np.dot(self.x[:-1], self.Rs.T)[:, None, :]     # past states
        log_Ps = log_Ps + self.r                                       # bias
        log_Ps = np.tile(log_Ps, (1, self.K, 1))                       # expand
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize
    
    def forward_pass(self,pi0,
                     Ps,
                     log_likes,
                     alphas):

        T = log_likes.shape[0]  # number of time steps
        K = log_likes.shape[1]  # number of discrete states

        assert Ps.shape[0] == T-1 or Ps.shape[0] == 1
        assert Ps.shape[1] == self.K
        assert Ps.shape[2] == self.K
        assert alphas.shape[0] == T
        assert alphas.shape[1] == K

        # Check if we have heterogeneous transition matrices.
        # If not, save memory by passing in log_Ps of shape (1, K, K)
        hetero = (Ps.shape[0] == T-1)
        alphas[0] = np.log(pi0) + log_likes[0]
        for t in range(T-1):
            m = np.max(alphas[t])
            alphas[t+1] = np.log(np.dot(np.exp(alphas[t] - m), Ps[t * hetero])+1e-32) + m + log_likes[t+1]
        return logsumexp(alphas[T-1])


    def _hmm_filter(self,Ps, ll):
        T, K = ll.shape

        # Forward pass gets the predicted state at time t given
        # observations up to and including those from time t
        alphas = np.zeros((T, K))
        self.forward_pass(self.pi0, Ps, ll, alphas)

        # Check if using heterogenous transition matrices
        hetero = (Ps.shape[0] == T-1)

        # Predict forward with the transition matrix
        pz_tt = np.empty((T-1, K))
        pz_tp1t = np.empty((T-1, K))
        for t in range(T-1):
            m = np.max(alphas[t])
            pz_tt[t] = np.exp(alphas[t] - m)
            pz_tt[t] /= np.sum(pz_tt[t])
            pz_tp1t[t] = pz_tt[t].dot(Ps[hetero*t])

        # Include the initial state distribution
        # Numba's version of vstack requires all arrays passed to vstack
        # to have the same number of dimensions.
        pi0 = np.expand_dims(self.pi0, axis=0)
        pz_tp1t = np.vstack((pi0, pz_tp1t))

        # Numba implementation of np.sum does not allow axis keyword arg,
        # and does not support np.allclose, so we loop over the time range
        # to verify that each sums to 1.
        for t in range(T):
            assert np.abs(np.sum(pz_tp1t[t]) - 1.0) < 1e-8

        return pz_tp1t

    
    def forward(self, x, input):
        return np.matmul(self.Cs[None, ...], x[:, None, :, None])[:, :, :, 0] \
            + np.matmul(self.Ds[None, ...], input[:, None, :, None])[:, :, :, 0] \
            + self.ds
            
    def _log_mean(self, x):
        return np.exp(x) * self.bin_size

    def _softplus_mean(self, x):
        return softplus(x) * self.bin_size
    
    def _ReLu(self, x):
        return np.array([_x*self.bin_size if _x > 0 else 0 \
                         for _x in np.squeeze(x)])

    def _log_link(self, rate):
        return np.log(rate) - np.log(self.bin_size)

    def _softplus_link(self, rate):
        return inv_softplus(rate / self.bin_size)
    
    ##### Dynamics stuff
    
    def _compute_mus(self, data, input):
        # assert np.all(mask), "ARHMM cannot handle missing data"
        K, M = self.K, self.M
        T, D = data.shape
        lags = 1 # self.lags
        As, bs, Vs, mu0s = self.As, np.zeros((self.K,self.D)), self.Bs, np.vstack((np.zeros(self.D),np.ones((D,D))))

        # Instantaneous inputs
        mus = np.empty((K, T, D))
        mus = []
        for k, (A, b, V, mu0) in enumerate(zip(As, bs, Vs, mu0s)):
            # Initial condition
            mus_k_init = mu0 * np.ones((lags, D))

            # Subsequent means are determined by the AR process
            mus_k_ar = np.dot(input[lags:, :M], V.T)
            for l in range(lags):
                Al = A[:, l*D:(l + 1)*D]
                mus_k_ar = mus_k_ar + np.dot(data[lags-l-1:-l-1], Al.T)
            mus_k_ar = mus_k_ar + b

            # Append concatenated mean
            mus.append(np.vstack((mus_k_init, mus_k_ar)))

        return np.array(mus)
    
    def log_likelihoods(self, data, input):
        
        L = 1 #self.lags
        mus = self._compute_mus(data,input) # x

        # Compute the likelihood of the initial data and remainder separately
        # stats.multivariate_studentst_logpdf supports broadcasting, but we get
        # significant performance benefit if we call it with (TxD), (D,), (D,D), and (,)
        # arrays as inputs
        
        ll_init = np.column_stack([stats.diagonal_gaussian_logpdf(data[:L], mu[:L], sigmasq)
                               for mu, sigmasq in zip(mus, self.sigmasq_init)])

        ll_ar = np.column_stack([stats.diagonal_gaussian_logpdf(data[L:], mu[L:], sigmasq)
                               for mu, sigmasq in zip(mus, self.sigmasq_init)])


        # Compute the likelihood of the initial data and remainder separately
        return np.row_stack((ll_init, ll_ar))
    
    ###### Emissions stuff
    def emmisions_log_likelihoods(self, data, input, x):
        assert data.dtype == int
        lambdas = self._log_mean(self.forward(x, input))
        mask = np.ones_like(data, dtype=bool)
        lls = -gammaln(data[:,None,:] + 1) -lambdas + data[:,None,:] * np.log(lambdas+1e-8)
        return np.sum(lls * mask[:, None, :], axis=2)
    
    def invert(self, data, input=None):
        if self.bin_size < 1:
            # sigma = 0.035
            # window = make_halfgauss_kern(sigma, self.bin_size)
            # yhat = smooth(data,window)
            yhat = moveavg(data,20)
        else:
            # sigma = 0.035
            # window = make_halfgauss_kern(sigma, self.bin_size)
            # yhat = smooth(data,window)
            yhat=moveavg(data,5)
        # Only invert observable, non-augmented dimension
        xhat = self._softplus_link(np.clip(yhat, 0.01, np.inf))
        xhat = self._invert(xhat, input=input)
        xhat = moveavg(xhat,10)
        #xhat = smooth(xhat,window)
        
        # DAW Commented
        if self.bin_size < 1:
            xhat = np.clip(xhat, -0.95, 0.95)
    
        # in all models, x starts in between boundaries at [-1,1]
        if np.abs(xhat[0]).any()>1.0:
            xhat[0] = 0.05*npr.randn(1,self.D)
        return xhat
    
    def hess(self,x,y,u):
        # For stability, we avoid evaluating terms that look like exp(x)**2.
        # Instead, we rearrange things so that all terms with exp(x)**2 are of the form
        # (exp(x) / exp(x)**2) which evaluates to sigmoid(x)sigmoid(-x) and avoids overflow.
        lambdas = self._softplus_mean(self.forward(x, u))[:, 0, :] / self.bin_size + 1e-20
        linear_terms = -np.dot(x,self.Cs[0].T)-np.dot(u,self.Ds[0].T)-self.ds[0]
        expterms = np.exp(linear_terms)
        outer = logistic(linear_terms) * logistic(-linear_terms)
        diags = outer * (y / lambdas - y / (lambdas**2 * expterms) - self.bin_size)
        return np.einsum('tn, ni, nj ->tij', diags, self.Cs[0], self.Cs[0])
    
    def jac(self,x,u):
        lambdas = self._softplus_mean(self.forward(x, u))[:, 0, :] + 1e-20
        return 1/lambdas*logistic(self.forward(x, u))[:, 0, :]*self.bin_size @ self.Cs[0]
    
    def _invert(self, data, input=None):
        """
        Approximate invert the linear emission model with the pseudoinverse

        y = Cx + d + noise; C orthogonal.
        xhat = (C^T C)^{-1} C^T (y-d)
        """
        # Invert with the average emission parameters
        C = np.mean(self.Cs, axis=0)
        D = np.mean(self.Ds, axis=0)
        d = np.mean(self.ds, axis=0)
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

        # Account for the bias
        if input is not None:
            bias = input.dot(D.T) + d
        else:
            bias = d

        # Project data to get the mean
        return (data - bias).dot(C_pseudoinv)
    
    def get_h_obs(self, data, input, x, invert=False):
        if invert:
            self.h_obs = (1.0 / self.initial_variance) * self. \
                invert(data, input=input)
        else:
            self.h_obs = self.J_obs @ x
    
    def step(self, u, y):
        #assert(np.isfinite(np.nan))
        self.hmm_filter(self.u,self.y,self.x)
        #self._kalman_info_filter(u,y)
        self._ExPKF_filter(u, y)
    
    def hmm_filter(self, u, y, x):
        if self.x.shape[0] > 1:
            Ps = np.exp(self.log_transition_matrices())
            ll = self.log_likelihoods(x, u) # likelihoods of the predicted states given dynamics
            pz_tp1t = self._hmm_filter( Ps, ll)
            #assert(np.isfinite(np.nan))
            self.zhat = np.argmax(pz_tp1t[-1])
            self.zhat_arr.append(copy.deepcopy(self.zhat))
        else:
            self.zhat = 0
    
    def _ExPKF_filter(self,u,y):
        """
        Extended Poisson Kalman filter for time-varying linear dynamical system with inputs.
        Following 2nd order Taylor expansion from Santitissadeekorn et al. 2020. 
        
        lambda = exp{Cx + Du + d}
        
        d log lambda/dx = C
        d^2 log lambda/dx^2 = 0
        
        P^-1_{k|k} = P^-1_{k|k-1} + C * C^T * exp{ Cx + Du + d}
        """
       
        # 1. Predict
        self.u = np.vstack([self.u,u])
        self.y = np.vstack([self.y,y])
        x_pred = self.As[self.zhat] @ self.x[-1] + self.Bs[self.zhat] @ self.u[-1] # = x_{k | k-1}
        self.x_pred = np.vstack([self.x_pred,x_pred])
        if not self.constant:
            P_pred = self.As[self.zhat] @ self.P[-1] @ self.As[self.zhat].T + self.Q # = P_{k | k-1}
        else:
            Px_pred = self.As[self.zhat] @ self.P[-1][:self.D,:self.D] @ self.As[self.zhat].T \
                + self.Q[:self.D,:self.D]
            Palph_pred = self.P[-1][self.D:,self.D:] + self.Q[self.D:,self.D:]
            P_pred = block_diag(Px_pred,Palph_pred)
        #P_pred =  self.P[-1] + self.Q # = P_{k | k-1}
        self.P_pred = np.concatenate([self.P_pred,P_pred[None,:,:]],axis=0)
        self.P_inv = np.linalg.inv(self.P_pred[-1])
        self.x = np.vstack([self.x,x_pred]) 
        
        # 2. Update
        arg = self.forward(self.x[-1][None,:], self.u[-1][None,:])[:, 0, :]
        # Using lambda = ReLu(Cx + Du + d)
        if self.constant == 'relu':
            lambd = self._ReLu(arg) / self.bin_size
            innovation = y - lambd * self.bin_size
            factor = lambd[:,None]*self.bin_size*(arg>0).T
            first_term = (1/arg * (self.Cs[0]*(arg>0).T)).T @ (1/arg * (self.Cs[0]*factor))
            second_term = -(1/arg * (self.Cs[0]*(arg>0).T)).T @ (1/arg * (self.Cs[0]*innovation[:,None]*(arg>0).T))
            self.P_inv = self.P_inv + first_term - second_term
            self.P = np.concatenate([self.P,np.linalg.inv(self.P_inv)[None,:,:]],axis=0)
            x_filt = x_pred + np.squeeze(self.P[-1][:self.D,:self.D] @ (1/arg @ (self.Cs[0]*innovation[:,None])).T)
            self.x[-1] = x_filt
        elif self.constant == 'softplus':
            lambd = self._softplus_mean(self.forward(self.x[-1][None,:], self.u[-1][None,:]))[0,0,:] /self.bin_size
            innovation = y - lambd * self.bin_size
            jac = self.jac(self.x[-1][None,:], self.u[-1][None,:])
            hess = self.hess(self.x[-1][None,:], y, self.u[-1][None,:])
            #assert np.isfinite(np.nan)
            self.P_inv = self.P_inv + np.sum(np.tile(jac[None,:,:],(self.Cs.shape[1],1,1)) * lambd[:,None,None] * self.bin_size - \
                                              innovation[:,None,None] * np.tile(hess,(self.Cs.shape[1],1,1)),axis=0)  
            self.P = np.concatenate([self.P,np.linalg.inv(self.P_inv)[None,:,:]],axis=0)
            x_filt = x_pred + np.squeeze(self.P[-1][:self.D,:self.D] @ np.sum(self.Cs[0].T*innovation[None,:],axis=1))
            self.x[-1] = x_filt 
        elif self.constant == 'softplus1':
            lambd = self._softplus_mean(self.forward(self.x[-1][None,:], self.u[-1][None,:]))[0,0,:] /self.bin_size
            lambd = np.clip(lambd,0.01,100)
            dlambd = logistic(arg)
            self.P_inv = self.P_inv - 1/lambd**2 * logistic(arg)*(y - 1) @ self.Cs[0]+\
                (self.Cs[0].T*1/lambd*logistic(arg)*(1-logistic(arg))*(y-1))@self.Cs[0]
            self.P = np.concatenate([self.P,np.linalg.inv(self.P_inv)[None,:,:]],axis=0)
            
            x_filt = x_pred + (self.P[-1] @ (1/lambd*logistic(arg)*(y - 1) @ self.Cs[0]).T).T
            self.x[-1] = x_filt 
        # Using lambda = exp{Cx + Du + d}
        elif not self.constant:
            #lambd = np.exp(arg)[0]
            lambd = self._softplus_mean(self.forward(self.x[-1][None,:], self.u[-1][None,:]))[0,0,:] /self.bin_size
            innovation = y - lambd * self.bin_size
            C_sq = self.Cs[0].T @ (self.Cs[0] * lambd[:,None] * self.bin_size)
            self.P_inv = self.P_inv + C_sq
            self.P = np.concatenate([self.P,np.linalg.inv(self.P_inv)[None,:,:]],axis=0)
            x_filt = x_pred + np.squeeze(self.P[-1][:self.D,:self.D] @ np.sum(self.Cs[0].T*innovation[None,:],axis=1))
            self.x[-1] = x_filt 
            
            
            # lambd = np.exp(arg)[0]
            ## This looks really good for control but I don't think it's real
            ## filter doesn't change with x_variance. I think what's happening
            ## here is that the model is very good, so the predictions are good
            ## and then there's no update. Can get equivalently smooth estimates
            ## By setting x_variance very low w/ above code
            # lambd = self._softplus_mean(self.forward(self.x[-1][None,:], self.u[-1][None,:]))[0,0,:] /self.bin_size
            # innovation = y - lambd * self.bin_size
            # C_sq = self.Cs[0].T @ self.Cs[0] 
            # self.P_inv = self.P_inv + np.sum(np.tile(C_sq[None,:,:],(self.Cs.shape[1],1,1)) * lambd[:,None,None] * self.bin_size ,axis=0)# \
            #     # - innovation * (self.Cs[0].T * (1/arg)[0,:] @ self.Cs[0])
            # self.P = np.concatenate([self.P,np.linalg.inv(self.P_inv)[None,:,:]],axis=0)
            # x_filt = x_pred + self.P[-1][:self.D,:self.D] @ self.Cs[0].T @ innovation
            # self.x[-1] = x_filt 
       
        else:
        # # Using lambda = alpha * exp{Cx+ Du + d}
        
            lambd = self.alphas * np.exp(arg)[0]
            lambd = self._softplus_mean(self.forward(self.x[-1][None,:], self.u[-1][None,:]))[0,0,:] /self.bin_size
            innovation = y - lambd * self.bin_size
            C_sq = np.hstack((self.Cs[0], (1/self.alphas)*np.eye(self.N))).T @ \
                np.hstack((self.Cs[0], (1/self.alphas)*np.eye(self.N)))
            second_deriv = np.vstack((np.zeros((self.D,self.D+self.N)),\
                                      np.hstack((np.zeros((self.N,self.D)),(1/self.alphas**2)*np.eye(self.N))))) 
            self.P_inv = self.P_inv + np.sum(np.tile(C_sq[None,:,:],(self.Cs.shape[1],1,1)) * lambd[:,None,None] * self.bin_size - \
                                              innovation[:,None,None] * np.tile(second_deriv,(self.Cs.shape[1],1,1)),axis=0) 
    
            # C_sq = self.Cs[0] @ self.Cs[0].T
            # self.P_inv = self.P_inv + np.sum(C_sq @ (lambd[:,None] * self.bin_size) ,axis=0)
            theta_prev = np.hstack((x_pred,self.alphas))
            self.P = np.concatenate([self.P,np.linalg.inv(self.P_inv)[None,:,:]],axis=0)
            
            theta = theta_prev + self.P[-1] @ np.hstack((self.Cs[0], (1/self.alphas)*np.eye(self.N))).T @ innovation
            #x_filt = x_pred + self.P[-1][:self.D,:self.D] @ self.Cs[0].T @ innovation
            x_filt = theta[:self.D]
            self.alphas = theta[self.D:]
            self.x[-1] = x_filt 
            # factor = lambd[:,None] * self.bin_size
            # first_term = np.hstack((self.Cs[0]*factor, (1/self.alphas)*np.eye(self.N))).T @ \
            #     np.hstack((self.Cs[0], (1/self.alphas)*np.eye(self.N)))
            # second_deriv = np.vstack((np.zeros((self.D,self.D+self.N)),\
            #                           np.hstack((np.zeros((self.N,self.D)),(1/self.alphas**2)*np.eye(self.N))))) 
            # #self.P_inv = self.P_inv + np.sum(np.tile(C_sq[None,:,:],(self.Cs.shape[1],1,1)) * lambd[:,None,None] * self.bin_size - \
            # summer = np.zeros_like(self.P_inv)
            # for i in range(len(lambd)):
            #     summer += innovation[i] * second_deriv
            # self.P_inv = self.P_inv + first_term - summer
    
            # # C_sq = self.Cs[0] @ self.Cs[0].T
            # # self.P_inv = self.P_inv + np.sum(C_sq @ (lambd[:,None] * self.bin_size) ,axis=0)
            # theta_prev = np.hstack((x_pred,self.alphas))
            # self.P = np.concatenate([self.P,np.linalg.inv(self.P_inv)[None,:,:]],axis=0)
            
            # theta = theta_prev + self.P[-1] @ np.hstack((self.Cs[0], (1/self.alphas)*np.eye(self.N))).T @ innovation
            # #x_filt = x_pred + self.P[-1][:self.D,:self.D] @ self.Cs[0].T @ innovation
            # x_filt = theta[:self.D]
            # self.alphas = theta[self.D:]
            # self.x[-1] = x_filt 
        assert np.all(x_filt < 1e8)
        
        
    def filter_trial(self, u_vec, y_vec):
        for inp, out in zip(u_vec,y_vec):
            ti = time.perf_counter()
            self.step(inp,out)
            self.times.append(time.perf_counter()-ti)
            
def causal_vi(model,u,y):
    xhat = np.zeros((len(y),model.D))
    zhat = np.zeros(len(y))
    Sigmas_hat = np.zeros((len(y),model.D,model.D))
    times = []
    for t in range(2,len(y)):
        ti = time.time()
        elbos, posterior = model.approximate_posterior(y[:t], u[:t], None, None,num_iters=1,verbose=0)
        times.append(time.time()-ti)
        # Estimate states for optimal control
        xhat[t] = posterior.mean_continuous_states[0][-1]
        Sigmas_hat[t] = posterior.continuous_expectations[0][2][-1]
        #yhat = model.smooth(xhat, y, input=u)
        zhat[t] = np.argmax(posterior.discrete_expectations[0][0][-1])
        
    return xhat, zhat, Sigmas_hat, times