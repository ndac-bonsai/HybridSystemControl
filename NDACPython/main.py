import numpy as np
import numpy.random as npr

from ssm.lds import SLDS
from typing import Any

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import partial

from scipy.stats import multivariate_normal
import ssm.stats as stats

from ssm.util import logistic, logit, softplus, inv_softplus
from autograd.scipy.special import logsumexp, gammaln
import numpy as np
import copy
import numpy.random as npr
from scipy.linalg import block_diag
import time

class OnlineSLDSModel(SLDS):

    def __init__(self,
                    N: int,
                    D: int,
                    K: int,
                    M: int,
                    transitions: str,
                    #transition_kwargs: dict,
                    dynamics: str,
                    #dynamics_kwargs: dict,
                    emissions: str,
                    #emissions_kwargs: dict,
                    dt: float
                    ) -> None:
        self.N = N
        self.D = D
        self.K = K
        self.M = M
        self.dt = dt
        self.x = 0.05*npr.randn(1,self.D)#.astype(np.float32)
        self.z = 0 #.astype(np.float32)
        self.y = np.zeros((1,self.M)).astype(np.float64)
        
        self.batch = None
        self.is_running = False
        self._optimization_finished = False
        self.loop = None
        self.thread = None

        super().__init__(N, K=K, D=D, M=M,
                            transitions=transitions,
                            #transition_kwargs=transition_kwargs,
                            dynamics=dynamics,
                            #dynamics_kwargs=dynamics_kwargs,
                            emissions=emissions,
                            #emissions_kwargs=emissions_kwargs
                            )
        
        # Get model parameters
        self.As = self.dynamics.As.flatten()
        self.Bs = self.dynamics.Vs.flatten()
        self.Cs = self.emissions.Cs.flatten()
        self.Fs = self.emissions.Fs.flatten()
        self.ds = self.emissions.ds.flatten()
        self.Sigmas_init = self.dynamics.Sigmas_init.flatten()
        self.Sigmas = self.dynamics.Sigmas.flatten()
        scale = 200;
        try:
            self.Rs = self.transitions.Rs.flatten()
        except:
            self.Rs = np.vstack((np.zeros(D),scale*np.eye(D))).flatten()
        try:
            self.r = self.transitions.r.flatten()
        except:
            self.r = np.concatenate(([0],-scale*np.ones(D)))
        self.pi0 = self.init_state_distn.initial_state_distn
        self.alpha_variance = 1e-3
        self.x_variance = np.diag(self.dynamics.Sigmas[0])
        self.constant = False
        
    def sample(self, u, with_noise=True):
        # Convert the input into a numpy array
        u = [point for point in u]
        u = np.asarray(u)
        u = u[:,None]
        
        z = np.zeros((1,)).astype(int)
        z[0] = self.z

        N = self.N
        K = self.K
        D = (self.D,) if isinstance(self.D, int) else self.D
        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)
        mask = np.ones((1,)+D)
        tag = None
        
        # Find transition probabilities given the current state and input
        Pt = np.exp(self.transitions.log_transition_matrices(self.x, u, mask, tag))[0]
        # Choose Regime according to transition probabilities
        self.z = npr.choice(self.K, p = Pt[z[0]])
        # Update state by sampling from the dynamics distribution
        self.x = self.dynamics.sample_x(z[0], self.x, input=u, tag=tag, with_noise=with_noise)[0]

        # Sample emissions from latent states
        self.y = self.emissions.sample(z, self.x[None,:], input=u.T, tag=tag)[0].astype(np.float64)
            
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
        self.x_variance = np.array(x_variance)
        self.alpha_variance = alpha_variance
        if type(x_variance) == float:
            self.initial_variance = np.hstack((self.x_variance*np.ones((1,self.D)), \
                                               self.alpha_variance*np.ones((1,self.N))))
        else:
            self.initial_variance = np.hstack((self.x_variance[None,:], \
                                               self.alpha_variance*np.ones((1,self.N))))  
        self.bin_size = bin_size
        
        self.Qs_inv = [np.linalg.inv(Q) for Q in Qs]
        self.As_inv = [np.linalg.inv(A) for A in As]

        
    def initialize(self):

        #self.x = self.J_pred @ self.h_pred
        self.xhat = 0.05*npr.randn(1,self.D) # inititalize x near 0
        self.x_pred = np.zeros_like(self.xhat)
        
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

    
    ### Transitions/HMM things
    def log_transition_matrices(self):
        #log_Ps = np.dot(input[1:], self.Ws.T)[:, None, :]              # inputs
        log_Ps =  np.dot(self.xhat[:-1], self.Rs.T)[:, None, :]     # past states
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
                               for mu, sigmasq in zip(mus, self.sigmasq)])


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
            yhat = moveavg(data,20)
        else:
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
    
    
    def step(self, u, y):
        #assert(np.isfinite(np.nan))
        self.hmm_filter(self.u,self.y,self.xhat)
        #self._kalman_info_filter(u,y)
        self._ExPKF_filter(u, y)
    
    def hmm_filter(self, u, y, x):
        if self.xhat.shape[0] > 1:
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
        x_pred = self.As[self.zhat] @ self.xhat[-1] + self.Bs[self.zhat] @ self.u[-1] # = x_{k | k-1}
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
        self.xhat = np.vstack([self.xhat,x_pred]) 
        
        # 2. Update
        arg = self.forward(self.xhat[-1][None,:], self.u[-1][None,:])[:, 0, :]
        
        # Using lambda = exp{Cx + Du + d}
        
        #lambd = np.exp(arg)[0]
        lambd = self._softplus_mean(self.forward(self.xhat[-1][None,:], self.u[-1][None,:]))[0,0,:] /self.bin_size
        innovation = y - lambd * self.bin_size
        C_sq = self.Cs[0].T @ (self.Cs[0] * lambd[:,None] * self.bin_size)
        self.P_inv = self.P_inv + C_sq
        self.P = np.concatenate([self.P,np.linalg.inv(self.P_inv)[None,:,:]],axis=0)
        #x_filt = x_pred + np.squeeze(self.P[-1][:self.D,:self.D] @ np.sum(self.Cs[0].T*innovation[None,:],axis=1))
        x_filt = x_pred + np.squeeze(self.P[-1][:self.D,:self.D] @ np.sum(self.Cs[0].T*innovation,axis=1))

        self.xhat[-1] = x_filt 

        
        assert np.all(x_filt < 1e8)
  

class OnlineHMMKFEstimator(HMMKFEstimator):

        def __init__(self,
                        N: int,
                        D: int,
                        K: int,
                        M: int,
                        As: np.ndarray[Any, np.dtype[np.float64]],
                        Bs: np.ndarray[Any, np.dtype[np.float64]],
                        Cs: np.ndarray[Any, np.dtype[np.float64]],
                        Fs: np.ndarray[Any, np.dtype[np.float64]],
                        ds: np.ndarray[Any, np.dtype[np.float64]],
                        sigmasq_init: np.ndarray[Any, np.dtype[np.float64]],
                        Qs: np.ndarray[Any, np.dtype[np.float64]],
                        Rs: np.ndarray[Any, np.dtype[np.float64]],
                        r: np.ndarray[Any, np.dtype[np.float64]],
                        pi0: np.ndarray[Any, np.dtype[np.float64]],
                        alpha_variance: float,
                        x_variance: np.ndarray[Any, np.dtype[np.float64]],
                        dt: float,
                        constant: bool
                        ) -> None:
            
            As = np.reshape(As,(K,D,D))
            Bs = np.reshape(Bs,(K,D,M))
            Cs = np.reshape(Cs,(1,N,D))
            Ds = np.reshape(Fs,(1,N,M))
            ds = np.reshape(ds,(1,N))
            sigmasq_init = np.reshape(sigmasq_init,(K,D,D))
            Qs = np.reshape(Qs,(K,D,D))
            Rs = np.reshape(Rs,(K,D))
            pi0 = np.array(pi0)
            super().__init__(As, Bs, Cs, Ds, ds, sigmasq_init, Qs, Rs, r, pi0, 
                 alpha_variance=alpha_variance,x_variance=x_variance,bin_size=dt,constant=False)
            
            self.initialize()
            self.dt = self.bin_size
            self.x = 0.05*npr.randn(1,self.D)#.astype(np.float32)
            
            self.z = self.zhat #.astype(np.float32)
            self.y = np.zeros((1,self.N)).astype(np.float64)
            self.Sigmas = Qs
            self.sigmasq = np.tile(np.array([1e-3,1e-4,1e-4])[:,None],(1,self.D))
            self.sigmasq_init = np.tile(np.array([2e-3,1e-4,1e-4])[:,None],(1,self.D))
            #self.sigmasq = np.power(self.Sigmas,2)
            #self.sigmasq_init = self.sigmasq_init[:,:,0] + 1e-8
            self.batch = None
            self.is_running = False
            self._optimization_finished = False
            self.loop = None
            self.thread = None
        
        def filt(self, u, y):
            u = [point for point in u]
            u = np.asarray(u)
            u = u[:,None]

            y = [point for point in y]
            y = np.asarray(y)
            y = y[None,:]
            
            self.clear_history()
            
            self.step(u,y)
            self.x = self.xhat[-1]
            self.z = int(self.zhat_arr[-1])
            self.Sigmas = self.P_inv
            
        def clear_history(self):
            if self.xhat.shape[0] > 50:
                self.u = self.u[-50:]
                self.y = self.y[-50:]
                self.xhat = self.xhat[-50:]
                self.x_pred = self.x_pred[-50:]
                self.P_pred = self.P_pred[-50:]
                self.zhat_arr = self.zhat_arr[-50:]
                
class OnlineEstimator(SLDS):

        def __init__(self,
                        N: int,
                        D: int,
                        K: int,
                        M: int,
                        transitions: str,
                        #transition_kwargs: dict,
                        dynamics: str,
                        #dynamics_kwargs: dict,
                        emissions: str,
                        #emissions_kwargs: dict,
                        alpha_variance: float,
                        x_variance: np.ndarray[Any, np.dtype[np.float64]],
                        dt: float,
                        constant: bool
                        ) -> None:

            self.N = N
            self.D = D
            self.K = K
            self.M = M
            self.dt = dt
            self.x = 0.05*npr.randn(1,self.D)#.astype(np.float32)
            self.z = 0 #.astype(np.float32)
            self.y = np.zeros((1,self.M)).astype(np.float64)
            
            self.batch = None
            self.is_running = False
            self._optimization_finished = False
            self.loop = None
            self.thread = None

            super().__init__(N, K=K, D=D, M=M,
                                transitions=transitions,
                                #transition_kwargs=transition_kwargs,
                                dynamics=dynamics,
                                #dynamics_kwargs=dynamics_kwargs,
                                emissions=emissions,
                                #emissions_kwargs=emissions_kwargs
                                )
            self.pi0 = self.init_state_distn.initial_state_distn
            self.constant = constant
            self.x_variance = np.array(x_variance)
            self.alpha_variance = alpha_variance
            if type(x_variance) == float:
                self.initial_variance = np.hstack((self.x_variance*np.ones((1,self.D)), \
                                                   self.alpha_variance*np.ones((1,self.N))))
            else:
                self.initial_variance = np.hstack((self.x_variance[None,:], \
                                                   self.alpha_variance*np.ones((1,self.N))))  
           
            self.initialize()
            
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
            self.xhat = 0.05*npr.randn(1,self.D) # inititalize x near 0
            self.x_pred = np.zeros_like(self.xhat)
            
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
        
        def step(self, u, y):
            #assert(np.isfinite(np.nan))
            self.hmm_filter(self.u,self.y,self.xhat)
            #self._kalman_info_filter(u,y)
            self._ExPKF_filter(u, y)
        
        def hmm_filter(self, u, y, x):
            if self.xhat.shape[0] > 1:
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
            x_pred = self.As[self.zhat] @ self.xhat[-1] + self.Bs[self.zhat] @ self.u[-1] # = x_{k | k-1}
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
            self.xhat = np.vstack([self.xhat,x_pred]) 
            
            # 2. Update

            #lambd = np.exp(arg)[0]
            lambd = self._softplus_mean(self.forward(self.xhat[-1][None,:], self.u[-1][None,:]))[0,0,:] /self.bin_size
            innovation = y - lambd * self.bin_size
            C_sq = self.Cs[0].T @ (self.Cs[0] * lambd[:,None] * self.bin_size)
            self.P_inv = self.P_inv + C_sq
            self.P = np.concatenate([self.P,np.linalg.inv(self.P_inv)[None,:,:]],axis=0)
            #x_filt = x_pred + np.squeeze(self.P[-1][:self.D,:self.D] @ np.sum(self.Cs[0].T*innovation[None,:],axis=1))
            x_filt = x_pred + np.squeeze(self.P[-1][:self.D,:self.D] @ np.sum(self.Cs[0].T*innovation,axis=1))

            self.xhat[-1] = x_filt 

            
            assert np.all(x_filt < 1e8)
            
        def filt(self, u, y):
            u = [point for point in u]
            u = np.asarray(u)
            u = u[:,None]

            y = [point for point in y]
            y = np.asarray(y)
            y = y[None,:]
            
            self.clear_history()
            
            self.step(u,y)
            self.x = self.xhat[-1]
            self.z = int(self.zhat_arr[-1])
            self.Sigmas = self.P_inv
            
        def clear_history(self):
            if self.xhat.shape[0] > 50:
                self.u = self.u[-50:]
                self.y = self.y[-50:]
                self.xhat = self.xhat[-50:]
                self.x_pred = self.x_pred[-50:]
                self.P_pred = self.P_pred[-50:]
                self.zhat_arr = self.zhat_arr[-50:]
            
def moveavg(x, window_sz,mode='same'):
    window = np.ones(window_sz)
    window = window[:x.shape[0]]
    window /= window_sz
    return np.apply_along_axis(np.convolve, 0, x, window, mode=mode)