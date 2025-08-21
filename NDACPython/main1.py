# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 10:18:22 2025

@author: dweiss38
"""
import numpy.random as npr

from ssm.lds import SLDS
#from ssmdm.accumulation import LatentAccumulation
from typing import Any
from ssm.util import logistic, logit, softplus, inv_softplus

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
from scipy import sparse
import osqp

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
                    alpha_variance: float,
                    x_variance: float,
                    SigmasScaler: float,
                    dt: float,
                    with_noise: str
                    ) -> None:

        self.N = N
        self.D = D
        self.K = K
        self.M = M
        self.dt = dt
        self.bin_size = dt
        self.x = 0.05*npr.randn(self.D)#.astype(np.float32)
        self.z = 0 #.astype(np.float32)
        self.y = np.zeros((1,self.N)).astype(np.float64)
        self.u_prev = np.zeros((1,self.M)).astype(np.float64)
        
        self.with_noise = with_noise

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
                            emissions_kwargs={'bin_size': dt}
                            )
        self.As = self.dynamics.As
        self.Bs = self.dynamics.Vs
        self.bs = self.dynamics.bs
        self.Cs = self.emissions.Cs
        self.Ds = self.emissions.Fs
        self.ds = self.emissions.ds

        self.pi0 = self.init_state_distn.initial_state_distn
        self.x_variance = x_variance
        self.alpha_variance = alpha_variance
        self.x_variance = x_variance
        self.alpha_variance = alpha_variance
        self.dynamics.Sigmas = self.dynamics.Sigmas*SigmasScaler
        if type(x_variance) == float:
            self.initial_variance = x_variance*np.ones((1,self.D))
        else:
            self.initial_variance = x_variance[None,:]
      
        if type(alpha_variance) == float:
            self.E = np.eye(self.N)*alpha_variance
        else:
            self.E = alpha_variance
                
        
        self.initialize()
        
        self.ssParams = {'As': self.As, 'Bs': self.Bs, 'bs': self.bs,
                                  'Cs': self.Cs, 'ds':self.ds}
        
    def sample(self, u):
        # Convert the input into a numpy array
        u = [point for point in u]
        u = np.asarray(u)
        u = u[None,:]
        
        # u = np.vstack((u,u))
        # #self.u_prev = u[-1]
        
        # # Repeat continuous state for recurrent_only transitions
        # x = np.tile(self.x,(2,1))
        x = np.squeeze(self.x)
        
        z = np.zeros((1,)).astype(int)
        z[0] = self.z

        D = (self.D,) if isinstance(self.D, int) else self.D
        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)
        mask = np.ones((1,)+D)
        tag = None
        
        # Find transition probabilities given the current state and input
        Pt = np.exp(self.transitions.log_transition_matrices(np.tile(x,(2,1)), np.tile(u,(2,1)), mask, tag))[0]
        # Choose Regime according to transition probabilities
        self.z = npr.choice(self.K, p = Pt[z[0]])
        # Update state by sampling from the dynamics distribution
        self.x = self.dynamics.sample_x(z[0], x[None,:], input=u[-1], tag=tag, with_noise=self.with_noise)
        # Sample emissions from latent states
        self.y = self.emissions.sample(z, self.x[None,:], input=u[-1,None], tag=tag)[0].astype(np.float64)
        
    def initialize(self):
        self.zhat = 0
        
        #self.x = self.J_pred @ self.h_pred
        self.xhat = 0.05*npr.randn(1,self.D) # inititalize x near 0
        self.x_pred = np.zeros_like(self.xhat)
        
        self.P = (self.initial_variance[0,:self.D][None,None,:] * self.dynamics.Sigmas)[0][None,:,:]
        self.Q = self.P[0]

        self.P_pred = np.zeros_like(self.P)
        self.u = np.zeros(self.M)
        self.y = np.zeros(self.N,dtype=int)
        self.zhat_arr = [0]
        self.alphas = 0.9 * np.ones(self.N)
    
    def forward_pass(self,pi0,
                     Ps,
                     log_likes,
                     alphas):

        T = log_likes.shape[0]  # number of time steps
        K = log_likes.shape[1]  # number of discrete states

        assert Ps.shape[0] == T-1 or Ps.shape[0] == 1
        assert Ps.shape[1] == K
        assert Ps.shape[2] == K
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
    
    def step(self, u, y):
        #assert(np.isfinite(np.nan))
        self.hmm_filter(self.u,self.y,self.xhat)
        #self._kalman_info_filter(u,y)
        self._ExPKF_filter(u, y)
    
    def hmm_filter(self, u, y, x):
        if self.xhat.shape[0] > 1:
            Ps = np.exp(self.transitions.log_transition_matrices(x,u,None,None))
            ll = self.dynamics.log_likelihoods(x, u, np.ones_like(x)) # likelihoods of the predicted states given dynamics
            pz_tp1t = self._hmm_filter( Ps, ll)
            #assert(np.isfinite(np.nan))
            self.zhat = np.argmax(pz_tp1t[-1])
            self.zhat_arr.append(copy.deepcopy(self.zhat))
        else:
            self.zhat = 0
            
    def jac(self,x,u):
        lambdas = self._softplus_mean(self.forward(x, u))[:, 0, :] / self.bin_size + 1e-20
        return (1/lambdas*logistic(self.forward(x, u))[:, 0, :])[0,:,None] * self.Cs[0]
    
    def hess(self,x,y,u):
        lambdas = self._softplus_mean(self.forward(x, u))[:, 0, :] / self.bin_size + 1e-20
        linear_terms = -np.dot(x,self.Cs[0].T)-np.dot(u,self.Ds[0].T)-self.ds[0]
        expterms = np.exp(linear_terms)
        outer = logistic(linear_terms) * logistic(-linear_terms)
        diags = outer * (y / lambdas - y / (lambdas**2 * expterms) - self.bin_size + self.bin_size/(lambdas * expterms))
        return np.einsum('tn, ni, nj ->tij', diags, self.Cs[0], self.Cs[0])

    def update(self,x,y,u):
        lambdas = self._softplus_mean(self.forward(x, u))[:, 0, :] / self.bin_size + 1e-20
        linear_terms = -np.dot(x,self.Cs[0].T)-np.dot(u,self.Ds[0].T)-self.ds[0]
        expterms = np.exp(linear_terms)
        outer = logistic(linear_terms) * logistic(-linear_terms)
        diags = outer * (y / (lambdas**2 * expterms) - y / lambdas + self.bin_size)
        return np.einsum('tn, ni, nj ->tij', diags, self.Cs[0], self.Cs[0])
    
    def forward(self, x, input):
        return np.matmul(self.Cs[None, ...], x[:, None, :, None])[:, :, :, 0] \
            + np.matmul(self.Ds[None, ...], input[:, None, :, None])[:, :, :, 0] \
            + self.ds
            
    def _softplus_mean(self, x):
        return softplus(x) * self.bin_size
    

    def _softplus_link(self, rate):
        return inv_softplus(rate / self.bin_size)
    
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
        #x_pred = self.sample(self.u[-1])
        x_pred = self.dynamics.sample_x(self.zhat,self.xhat[None,-1,:],input=self.u[-1],with_noise=False)
        #x_pred = self.As[self.zhat] @ self.xhat[-1] + self.Bs[self.zhat] @ self.u[-1] + self.bs[self.zhat] # = x_{k | k-1}
        #assert(False)
        self.x_pred = np.vstack([self.x_pred,x_pred])
        P_pred = self.As[self.zhat] @ self.P[-1] @ self.As[self.zhat].T + self.Q # = P_{k | k-1}
  
        self.P_pred = np.concatenate([self.P_pred,P_pred[None,:,:]],axis=0)
        self.P_inv = np.linalg.inv(self.P_pred[-1])
        self.xhat = np.vstack([self.xhat,x_pred]) 
        
        # 2. Update

        #lambd = np.exp(arg)[0]
        lambd = self._softplus_mean(self.forward(self.xhat[-1][None,:], self.u[-1][None,:]))[:,0,:] /self.bin_size
        innovation = y - lambd * self.bin_size
        #jac = self.jac(self.x[-1][None,:], self.u[-1][None,:])
        
        jac = self.jac(self.xhat[-1][None,:], self.u[-1][None,:])
        
        hess = self.hess(self.xhat[-1][None,:], y, self.u[-1][None,:])
        
        
        self.P_inv = self.P_inv + jac.T @ (jac * lambd[0,:,None] * self.bin_size) - \
                                         hess[0,:,:] 
        
        #self.P_inv = self.P_inv + self.update(self.xhat[-1][None,:], y, self.u[-1][None,:])[0,:,:] 

        self.P = np.concatenate([self.P,np.linalg.inv(self.P_inv)[None,:,:]],axis=0)

        x_filt = x_pred + (self.P[-1] @ (jac.T @ (self.E @ innovation.T)))[:,0]
        self.xhat[-1] = x_filt 
        
        assert np.all(x_filt < 1e8)
        
    def filt(self, u, y):
        u = [point for point in u]
        u = np.asarray(u)
        u = u[None,:]

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
            
    
        
#%% Controllers

class MPCController(object):
    def __init__(self, 
                 ssParams: dict, 
                 Q: list[float], 
                 R: list[float], 
                 Mopto: int,
                 N: int = 10, 
                 lb: float = 0,
                 ub: float = 1) -> None:
        """
        Initialize model parameters
        """
        xconstraint = None
        self.xconstraint = xconstraint

        self.As, self.Bs, self.bs, self.Cs, self.ds = [np.array(ssParams[k]) if not \
                                     isinstance(ssParams[k], int) else ssParams[k]
                                     for k in ssParams.keys()]
        D = self.As.shape[1]
        K = self.As.shape[0]
        self.Qs = np.array([Q*np.eye(D) for k in range(K)])
        self.Rs= np.array([R*np.eye(D) for k in range(K)])
        self.N, self.lb, self.ub, = N, lb, ub
            
        self.solver = OSQPSolver(self.As[0],self.Bs[0],self.bs[0],self.Cs[0],self.ds[0],
                                 self.Qs[0],self.Rs[0],N=N,lb=lb,ub=ub,xconstraint=xconstraint)
        self.u = np.zeros(self.Bs.shape[2])
        self.D = self.As.shape[0]

    def set_reference(self, xref):
        self.xref = xref
        
    def quad_form(self, vec, mat):
        return np.einsum('...i,...i->...', vec.dot(mat), vec)
        
    def get_finite_horizon(self, t):
        N = self.N+1
        if t+N < self.xref.shape[0]:
            return self.xref[t:t+N,:]#, self.uref[t:t+N,:]
        else:
            return np.vstack((self.xref[t:,:],np.tile(self.xref[-1,:],(t-self.xref.shape[0]+N,1))))
            #utmp = np.vstack((uref[t:,:],np.tile(uref[-1,:],(t-uref.shape[0]+N,1))))
    
    def ustar(self, xhat, zhat, t):
        self.xref = np.ones((100,self.D))
        if zhat != 0:
            self.solver = OSQPSolver(self.As[zhat],self.Bs[zhat],self.bs[zhat],self.Cs[0],self.ds[0],self.Qs[zhat],\
                                     self.Rs[zhat],N=self.N,lb=self.lb,ub=self.ub,xconstraint=self.xconstraint)
        x_fh = self.get_finite_horizon(t)
        return self.solver.solve(xhat,x_fh)
            
class OSQPSolver(object):
    '''
    @article{osqp,
      author  = {Stellato, B. and Banjac, G. and Goulart, P. and Bemporad, A. and Boyd, S.},
      title   = {{OSQP}: an operator splitting solver for quadratic programs},
      journal = {Mathematical Programming Computation},
      volume  = {12},
      number  = {4},
      pages   = {637--672},
      year    = {2020},
      doi     = {10.1007/s12532-020-00179-2},
      url     = {https://doi.org/10.1007/s12532-020-00179-2},
    }'''
    def __init__(self, A, B, b, C, d, Q, R, N=10, lb=0, ub=1,xconstraint=False):
        self.Ad, self.Bd, self.bd, self.C, self.d, self.Q, self.R, self.N, self.lb, self.ub, self.xconstraint = \
            sparse.csc_matrix(A), sparse.csc_matrix(B), b, C, d, Q, R, N, lb, ub, xconstraint
        
        if xconstraint:
            delta = np.log(1+np.exp(d))
            xmin = np.array([-delta])[0]
            xmax = 60*np.ones_like(delta)
        else:
            xmin = -np.inf*np.ones(A.shape[1])
            xmax = np.inf*np.ones(A.shape[1])
        #self.Cpseudoinv = np.linalg.inv(C.T@C) @ C.T
    
        self.nx, self.nu = B.shape
        
        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # Constraints
        umin = np.array([self.lb] * self.nu)
        umax = np.array([self.ub] * self.nu)
        
        # Static objectives
        self.QN = self.Q
        # - quadratic objective
        self.P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), self.QN,
                                sparse.kron(sparse.eye(N), R)], format='csc')
        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(self.nx)) + \
            sparse.kron(sparse.eye(N+1, k=-1), self.Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), \
                                              sparse.eye(N)]), self.Bd)
        Aeq = sparse.hstack([Ax, Bu])
        
        # - input and state constraints
        if xconstraint:
            Aineqx = np.kron(np.eye(self.N+1), self.C)
            Ainequ = np.eye(self.N*self.nu)
            Aineq = sparse.bsr_matrix(block_diag(Aineqx,Ainequ))
        else:
            Aineq = sparse.eye((N+1)*self.nx + N*self.nu)
        self.lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        self.uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
        
        # - OSQP constraints
        # A is a matrix s.t. l <= Ax <= u
        self.A = sparse.vstack([Aeq, Aineq], format='csc')
        #assert(False)
       
        
    
    def solve(self, x0, xref, time=False):
        '''
        
        Solve quadratic programming problems of the kind:
            1/2 x^T P x + q^T x
            s.t. l <= Ax <= u
        '''
        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # Constraints that depend on x0
        # - linear objective
        #q = np.hstack([np.kron(np.ones(N), -Q@xr.T), -QN@xr.T, np.zeros(N*nu)])
        
        q = np.hstack([(-self.Q@xref[:-1,:].T).T.flatten(),(-self.QN@xref[-1,:].T), \
                       np.zeros(self.N*self.nu)])
            
        # q = np.hstack([(-xref[:-1,:]@self.Q-(self.Q@xref[:-1,:].T).T).flatten(), \
        #                (-xref[-1,:]@self.QN-(self.QN@xref[-1,:].T)).flatten(), \
        #                np.zeros(self.N*self.nu)])
       
        # - input and state constraints
        #leq = np.hstack([-x0, np.zeros(self.N*self.nx)])
        leq = np.hstack([-x0, np.tile(self.bd,(self.N))]) # Changing this to incorporate state bias. self.bd is bias from the model
        ueq = leq
        
        # - OSQP constraints
        l = np.hstack([leq, self.lineq])
        u = np.hstack([ueq, self.uineq])
        '''
        H = block_diag(np.zeros((self.nx,self.nx)),self.H)
        q = 2*x0.T*np.vstack([np.zeros((self.nx,self.nx)),self.F.T])#np.array(self.F.todense()).T   
        #q = np.zeros_like(q)
        eps = 1.
        l = np.hstack([x0,self.lineq,xref.flatten()-eps])
        u = np.hstack([x0,self.uineq,xref.flatten()+eps])
        '''
        # Reference trajectory constraints
        
        
        #assert(False)
        # Create an OSQP object
        prob = osqp.OSQP()
        
        # Setup workspace
        prob.setup(self.P, q, self.A, l, u, verbose=False)
        #prob.setup(sparse.bsr_matrix(H), q, self.A, l, u, verbose=False)
        
    
        # Solve
        res = prob.solve()
        #assert(False)
        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')
        # return first control input
        if time:
            return res.x[-self.N*self.nu:-(self.N-1)*self.nu], res.info.run_time
        else:
            return res.x[-self.N*self.nu:-(self.N-1)*self.nu]


def moveavg(x, window_sz,mode='same'):
    window = np.ones(window_sz)
    window = window[:x.shape[0]]
    window /= window_sz
    return np.apply_along_axis(np.convolve, 0, x, window, mode=mode)


# class OnlineAccumulatorModel(SLDS,LatentAccumulation):
#     def __init__(self,
#                     N: int,
#                     D: int,
#                     K: int,
#                     M: int,
#                     transitions: str,
#                     #transition_kwargs: dict,
#                     dynamics: str,
#                     #dynamics_kwargs: dict,
#                     emissions: str,
#                     #emissions_kwargs: dict,
#                     alpha_variance: float,
#                     #x_variance: np.ndarray[Any, np.dtype[np.float64]],
#                     dt: float,
#                     constant: bool
#                     ) -> None:

#         self.N = N
#         self.D = D
#         self.K = K
#         self.M = M
#         self.dt = dt
#         self.bin_size = dt
#         self.x = 0.05*npr.randn(1,self.D)#.astype(np.float32)
#         self.z = 0 #.astype(np.float32)
#         self.y = np.zeros((1,self.M)).astype(np.float64)

#         self.batch = None
#         self.is_running = False
#         self._optimization_finished = False
#         self.loop = None
#         self.thread = None

#         LatentAccumulation.__init__(N, K=K, D=D, M=M,
#                             transitions=transitions,
#                             #transition_kwargs=transition_kwargs,
#                             dynamics=dynamics,
#                             #dynamics_kwargs=dynamics_kwargs,
#                             emissions=emissions,
#                             emissions_kwargs={'bin_size': dt}
#                             )
#         self.As = self.dynamics.As
#         self.Bs = self.dynamics.Vs
#         self.Cs = self.emissions.Cs
#         self.Ds = self.emissions.Fs
#         self.ds = self.emissions.ds

#         self.pi0 = self.init_state_distn.initial_state_distn
#         self.constant = constant
#         #self.x_variance = np.array(x_variance)
#         self.x_variance = np.diag(self.dynamics.Sigmas[0])
#         self.alpha_variance = alpha_variance
#         if type(self.x_variance) == float:
#             self.initial_variance = np.hstack((self.x_variance*np.ones((1,self.D)), \
#                                                self.alpha_variance*np.ones((1,self.N))))
#         else:
#             self.initial_variance = np.hstack((self.x_variance[None,:], \
#                                                self.alpha_variance*np.ones((1,self.N))))  
       
#         self.initialize()

    
        