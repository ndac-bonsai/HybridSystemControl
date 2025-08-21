# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 07:51:57 2024

@author: dweiss38
"""
import copy
import time
import control
import itertools
import numba

import osqp
from scipy import sparse
from scipy.linalg import block_diag

import numpy as np
import matplotlib.pyplot as plt


import scipy.optimize
from scipy.spatial import ConvexHull


#%% Controllers
        
class MPCController(object):
    def __init__(self, As, Bs, C, d, Q, R, Mopto, N=10, lb=0, ub=1, xconstraint=None):
        """
        Initialize model parameters
        """
        self.As, self.Bs, self.C, self.d, self.Q, self.R, self.Mopto, self.N, \
            self.lb, self.ub, self.xconstraint  =\
            As, Bs[:,:,-Mopto:], C, d, Q, R, Mopto, N, lb, ub, xconstraint
            
        self.solver = OSQPSolver(As[0],self.Bs[0],C,d,Q,R,N=N,lb=lb,ub=ub,xconstraint=xconstraint)
        self.u = np.zeros(Bs.shape[2])

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
        if zhat != 0:
            self.solver = OSQPSolver(self.As[zhat],self.Bs[zhat],self.C,self.d,self.Q,\
                                     self.R,N=self.N,lb=self.lb,ub=self.ub,xconstraint=self.xconstraint)
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
    def __init__(self, A, B, C, d, Q, R, N=10, lb=0, ub=1,xconstraint=False):
        self.Ad, self.Bd, self.C, self.d, self.Q, self.R, self.N, self.lb, self.ub, self.xconstraint = \
            sparse.csc_matrix(A), sparse.csc_matrix(B), C, d, Q, R, N, lb, ub, xconstraint
        
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
    
    def solve(self, x0, xref):
        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # Constraints that depend on x0
        # - linear objective
        #q = np.hstack([np.kron(np.ones(N), -Q@xr), -QN@xr, np.zeros(N*nu)])
        q = np.hstack([(-self.Q@xref[:-1,:].T).T.flatten(),(-self.QN@xref[-1,:].T), \
                       np.zeros(self.N*self.nu)])
            
        # - input and state constraints
        leq = np.hstack([-x0, np.zeros(self.N*self.nx)])
        ueq = leq
        
        # - OSQP constraints
        l = np.hstack([leq, self.lineq])
        u = np.hstack([ueq, self.uineq])
    
        # Create an OSQP object
        prob = osqp.OSQP()
        
        # Setup workspace
        prob.setup(self.P, q, self.A, l, u, verbose=False)
    
        # Solve
        res = prob.solve()
        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')
    
        # return first control input
        return res.x[-self.N*self.nu:-(self.N-1)*self.nu]


#%% Model Predictive Control functions

class ScipySolver:
  def __init__(self, N, Nu, lowerbound, upperbound):
    self.last_u = np.zeros(N*Nu)
    self.bounds = scipy.optimize.Bounds([lowerbound] * N*Nu, [upperbound] * N*Nu)

  def solve(self, H, h):
    def to_min(u):
      return 0.5 * u.T.dot(H).dot(u) + h.T.dot(u)

    def jac(u):
      return u.T.dot(H) + h.T

    def hess(u):
      return H

    res = scipy.optimize.minimize(to_min, self.last_u, method='trust-constr',
                                  jac=jac, hess=hess, bounds=self.bounds)
    self.last_u = res.x
    return res.x

def trajectory_optimization(model, xhat, zhat, xref, uref, t, Mphys, Q, R, N, lb, ub):
    # Get current dynamics by estimating current regime
    A = model.dynamics.As[zhat]
    B = model.dynamics.Vs[zhat][:,Mphys:]
    """
             B          0          ...   0
             AB         B          ...   0
             .          .       .        .
      calB = .          .          .     .
             .          .             .  .
             A^(N-1)B   A^(N-2)B   ...   B
    """
    calB = np.zeros((N * A.shape[0], N * B.shape[1]))
    vecc_gen = np.zeros((A.shape[0] * N, A.shape[0]))

    Ap = np.eye(A.shape[0])
    for i in range(N):
      for j in range(N - i):
        calB[A.shape[0] * (i + j) : A.shape[0] * (i + j + 1),
             B.shape[1] * j       : B.shape[1] * (j + 1)] = Ap.dot(B)
      Ap = A.dot(Ap)

      vecc_gen[A.shape[0] * i : A.shape[0] * (i + 1), :] = Ap

    # Build block diagonal cost matrices.
    calQ = np.kron(np.eye(N), Q)
    calR = np.kron(np.eye(N), R)
    calH = calB.T.dot(calQ).dot(calB) + calR
    calHinv = np.linalg.inv(calH)

    default_x_ref = np.zeros(calB.shape[0])
    default_u_ref = np.zeros(calB.shape[1]*B.shape[1])

    BtQ = calB.T.dot(calQ)

    def get_h(x, x_ref, u_ref):
      return BtQ.dot(vecc_gen.dot(x) - x_ref) - calR.dot(u_ref)

    def controller_inv(x, x_ref=default_x_ref, u_ref=default_u_ref):
      return calHinv[:B.shape[1], :].dot(-get_h(x, x_ref, u_ref))

    #solver = Solver(N,B.shape[1], lowerbound,upperbound)
    
    # def controller_opt(x, x_ref=default_x_ref, u_ref=default_u_ref):
    #   return solver.solve(calH, get_h(x, x_ref, u_ref))[:B.shape[1]]

    # return controller_inv, controller_opt
   
    Nu = uref.shape[1]
    Nx = xref.shape[1]
    bounds = scipy.optimize.Bounds([-np.inf] * N*Nx + [lb] * N*Nu,[np.inf] * N*Nx + [ub] * N*Nu)
    
    
    def quad_form(vec,mat):
        return np.einsum('...i,...i->...', vec.dot(mat), vec)
    
    def get_traj(x,u):
        return np.hstack((x.flatten(), u.flatten()))
        
    def get_finite_horizon(t):
        if t+N < xref.shape[0]:
            return xref[t:t+N,:], uref[t:t+N,:]
        else:
            xtmp = np.vstack((xref[t:,:],np.tile(xref[-1,:],(t-xref.shape[0]+N,1))))
            utmp = np.vstack((uref[t:,:],np.tile(uref[-1,:],(t-uref.shape[0]+N,1))))
            return xtmp, utmp
        
    x_fh, u_fh = get_finite_horizon(t)
    
    
    def _obj(traj):
        x = np.reshape(traj[:np.size(x_fh)],np.shape(x_fh))
        u = np.reshape(traj[np.size(x_fh):],np.shape(u_fh))
        xbar = x-x_fh
        ubar = u-u_fh
        return np.sum(quad_form(xbar,Q) + quad_form(ubar,R))
    
    def _obj_direct_shooting(u_flat):
        u = np.reshape(u_flat,u_fh.shape)
        x = np.zeros_like(x_fh)
        for i in range(1,N):
            x[i,:,None] = A @ x[i-1,:,None]+B @ u[i-1,:,None]
        return np.sum(quad_form(x-x_fh,Q) + quad_form(u-u_fh,R))
        
    # Slow and stupid
    trajectory = get_traj(x_fh, u_fh)
    # res = scipy.optimize.minimize(_obj, trajectory, method='trust-constr', bounds=bounds)
    # ustar = np.reshape(res.x[np.size(x_fh):],np.shape(u_fh))
    x_flat = trajectory[:np.size(x_fh)]
    u_flat = trajectory[np.size(x_fh):]
    solver = ScipySolver(N,B.shape[1], lb,ub)
    #bounds = scipy.optimize.Bounds([lb] * N*Nu,[ub] * N*Nu)

    #res = scipy.optimize.minimize(_obj_direct_shooting, u_flat, method='trust-constr', bounds=bounds)
    ustar = np.reshape(solver.solve(calH, get_h(xhat.T, x_flat, u_flat)),u_fh.shape)
    
    
    return ustar,x_fh,u_fh

    # def hess(u):
    #   return H
    
    # def simulate(xref, t, N)
  

    # res = scipy.optimize.minimize(to_min, self.last_u, method='trust-constr',
    #                               jac=jac, hess=hess, bounds=self.bounds)

#%% Solving ARE/LQR funcs
def lqr_fitK(A, B, Q, R,N=50):
    """
    Discrete-time linear quadratic regulator for a linear system.
 
    Compute the optimal feedback gain given a linear system, cost matrices
    Compute the control variables that minimize the cumulative cost.
    Solve for P using the dynamic programming method.
 
    :param Q: The state cost matrix
    :param R: The input cost matrix
    :param N: number of timesteps for finite time horizon lqr
 
    :return: K: Optimal feedback gain K for the state to obtain u* = K(x_error) 

    """
 
    # Solutions to discrete LQR problems are obtained using the dynamic 
    # programming method.
    # The optimal solution is obtained recursively, starting at the last 
    # timestep and working backwards.
    # You can play with this number
    
 
    # Create a list of N + 1 elements
    P = [None] * (N + 1)
     
    Qf = Q
 
    # LQR via Dynamic Programming
    P[N] = Qf
 
    # For i = N, ..., 1
    for i in range(N, 0, -1):
 
        # Discrete-time Algebraic Riccati equation to calculate the optimal 
        # state cost matrix
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
            R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)      
 
    # Create a list of N elements
    K = [None] * N
 
    # For i = 0, ..., N - 1
    for i in range(N):
 
        # Calculate the optimal feedback gain K
        K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
 
    return K


def lqr_fitKs(model, x_desired, Nlqr=100, augmented=False):
    D = model.D
    Demb = model.dynamics.D_emb
    K = model.K
    Mopto = model.dynamics.Mopto
    Mphys = model.dynamics.Mphys
    if x_desired[0] >= 1:
        Qacc1 = 50
        Qacc2 = 25
    else:
        Qacc2 = 50
        Qacc1 = 25
    Qemb = 1e-3
    if augmented:
        qs = np.tile(np.hstack((Qacc1,Qacc2,Qemb*np.ones((Demb,)))),(1,2))
    else:
        qs = np.hstack((Qacc1,Qacc2,Qemb*np.ones((Demb,))))
    Qs = np.array([np.eye(D)*qs for _ in range(K)])
    Rs = np.array([np.eye(Mopto)*25 for _ in range(K)])
    
    G = np.empty((K,Nlqr,Mopto,D))
    for i, (A,B,Q,R) in enumerate(zip(model.dynamics.As, model.dynamics.Vs[:,:,Mphys:], Qs, Rs)):
        #G[i,:,:], S[i,:,:], E[i,:] = control.lqr(A,B,Q,R)
        G[i,:,:,:] = lqr_fitK(A, B, Q, R, Nlqr)

    return G

def solveARE(A, B, Q, R,N=None,maxiter=500,tol=1e-3):

    # LQR via Dynamic Programming
    P = [Q]
    diff = [np.inf]
    # For i = N, ..., 1
    if N is not None:
        for i in range(1,N+1):
     
            # Discrete-time Algebraic Riccati equation to calculate the optimal 
            # state cost matrix
            P.append(Q + A.T @ P[i-1] @ A - (A.T @ P[i-1] @ B) @ np.linalg.pinv(
                R + B.T @ P[i-1] @ B) @ (B.T @ P[i-1] @ A))     
            diff.append(np.linalg.norm(P[i]-P[i-1]))
        return P.reverse(), diff.reverse()
    else:
        i = 1
        while(diff[-1] >= tol and i < maxiter):
            P.append(Q + A.T @ P[i-1] @ A - (A.T @ P[i-1] @ B) @ np.linalg.pinv(
                R + B.T @ P[i-1] @ B) @ (B.T @ P[i-1] @ A))     
            diff.append(np.linalg.norm(P[i]-P[i-1]))
            i += 1
        if i != maxiter:
            print('ARE converged after %d iterations, diff = %.5f' % (i, diff[-1]))
        else:
            print('WARNING: ARE tolerance %f not reached after %d iterations' % (tol, maxiter))
        return P[-1], diff[-1]
    
def getK(P,A,B,Q,R):
    return np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
#%% Reachability analysis funcs
def bounded_polyhedron(M,lb,ub):
    return np.array(list(itertools.product([lb, ub], repeat=M))).T

def unallowed_states(M,beta,d,lb=0,ub=1):
    '''
    M : Number of opto inputs
    lb : minimum opto actuation, should always be 0
    ub : maximum opto actuation, 1 as placeholder for maximum
    beta : matrix of opto effect on firing rates (N x M)
    d : baseline firing rate of neurons (N x 1)
    '''
    P = bounded_polyhedron(M, lb, ub)
    

def finite_time_Rw(A,B,P,N):
    Pstar = np.zeros((N,A.shape[0],P.shape[1]))
    for i in range(N):
        if i == 0:
            Pstar[i] = B @ P
        else:
            Pstar[i] = A**i @ B @ P
    Rw = np.sum(Pstar,axis=0)
    return Rw
        
def project_Rw(Rw, Dacc):
    projector = np.hstack([np.eye(Dacc), np.zeros((Dacc, Rw.shape[0]-Dacc))])
    return projector @ Rw
    
def Rw_convex_hull(Rw, Dacc=None):
    if Dacc is None:
        return ConvexHull(Rw.T)
    else:
        projection = project_Rw(Rw, Dacc)
        return ConvexHull(projection.T)
    
def get_conv_hull(A,B,N,Dacc,M,lb,ub):
    #P = bounded_polyhedron(M,lb,ub)
    X1,X2 = np.meshgrid(np.linspace(lb,ub),np.linspace(lb,ub))
    X1flat = X1.flatten()
    X2flat = X2.flatten()
    
    return finite_time_Rw(A,B,np.vstack([X1flat,X2flat]),N)
    # projection = project_Rw(hull, Dacc)
    # plt.figure()
    # plt.scatter(projection[0,:],projection[1,:])
    
def Gramian_dt(A,B,m):
    W_cd = control.ctrb(A,B) @ control.ctrb(A,B).T
    W_cd_m = np.zeros(A.shape)
    # Take A^0 out of loop because it returns ones, not eye(D)
    W_cd_m += B @ B.T
    for i in range(1,m):
        W_cd_m += A**m @ B @ B.T @ (A**m).T
    return W_cd, W_cd_m

def plot_Gramian(W_cd):
    l, v = np.linalg.eig(W_cd)
    vecs = v * l
    proj = vecs[:2,:]
    origin = np.zeros_like(proj) # origin point
    plt.quiver(*origin, proj[0,:], proj[1,:],scale = 2)
    plt.show()
    
def plot_reachable_set_eigs(A,B,N,Dacc,lb,ub):
    M = B.shape[1]
    hull = get_conv_hull(A,B,N,Dacc,M,lb,ub)
    pS1 = np.sum(np.logical_and(hull[0,:] >= 1, hull[1,:] < 1))/hull.shape[1]
    pS2 = np.sum(np.logical_and(hull[1,:] >= 1, hull[0,:] < 1))/hull.shape[1]
    m = np.mean(hull,axis=1)
    hull = hull - m[:,None]
    u,d,v = np.linalg.svd(hull)
    proj = u[:Dacc,:Dacc] * np.sqrt(d[:Dacc][None,:]) 
    #proj = np.cov(hull)[:2,:3]
    origin = np.tile(m[:2,None],(1,Dacc)) # origin point
    plt.quiver(*origin, proj[0,:], proj[1,:])
    plt.plot([1,1],[-1.5,1.5],linestyle='--',color='k')
    plt.plot([-1.5,1.5],[1,1],linestyle='--',color='k')
    #plt.show()
    return pS1, pS2
    
def phistar_inverse_image(A,B,k,P):
    if k == 0:
        return np.linalg.lstsq(B.T, P)[0]
    else:
        return np.linalg.lstsq(B.T @ (A**k).T, P)[0]
    #Q = B.T @ (A**k).T
    #pseudoinv = np.linalg.inv(Q.T @ Q) @ Q.T
    #return pseudoinv @ P
    
def phistar_image(A,B,k,S):
    if k == 0:
        return B.T @ S
    else:
        return B.T @ (A**k).T @ S

def reachable_set_m_step(A,B,m,lb,ub):
    '''
    Returns the reachable set of a system under polyhedral input constraints
    lb and ub. R(0,0,m) = x \in \sum_{k=0}^m A^k @ B @ phi*_k (phi*_k^-1(U)) 

    Parameters
    ----------
    A : system dynamics matrix (n x n)
    B : system input matrix (n x m)
    m : number of time steps to evaluate reachability
    lb : lower bound of input
    ub : upper bound of input

    Returns
    -------
    Reachable set as minimal set of points describing convex hull.

    '''
    M = B.shape[1]
    P = bounded_polyhedron(M,lb,ub)
    reachable_set = np.zeros((A.shape[0],P.shape[1]))
    for k in range(m):
        phi_inv = phistar_inverse_image(A,B,k,P)
        phi_phi_inv = phistar_image(A,B,k,phi_inv)
        if k == 0:
            reachable_set += B @ phi_phi_inv
        else:
            reachable_set += A**k @ B @ phi_phi_inv
    return reachable_set

def plot_region(P,Dacc,xlim=None,ylim=None):
    projection = project_Rw(P, Dacc).T
    hull = Rw_convex_hull(P,Dacc=Dacc)
    plt.figure()
    plt.scatter(projection[:,0],projection[:,1])
    plt.plot(projection[np.append(hull.vertices, hull.vertices[0]),0], 
             projection[np.append(hull.vertices, hull.vertices[0]),1], 'r--', lw=2)
    plt.fill(projection[hull.vertices,0], projection[hull.vertices,1], 'k', alpha=0.3)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
        
def plot_region_w_eigs(A,B,N,Dacc,M,lb,ub,xlim=None,ylim=None):
    P = reachable_set_m_step(A,B,M,lb,ub)
    plot_region(P,Dacc,xlim=xlim,ylim=ylim)
    pS1,pS2 = plot_reachable_set_eigs(A,B,N,Dacc,lb,ub)
    return pS1,pS2

#%% Under construction

def constrain(L_r, h_r, xN_r,lowerlim=0,upperlim=10):
    out = np.squeeze(-L_r @ xN_r - h_r.T)
    Lout = copy.deepcopy(L_r)
    hout = copy.deepcopy(h_r)
    Lout[out < lowerlim,:] = 0 
    hout[out < lowerlim] = lowerlim
    Lout[out > upperlim,:] = 0
    hout[out > upperlim] = -upperlim
    return Lout, hout

def augment_model(model,Q_m=None):
    # Make new model
    augmented_model = copy.deepcopy(model)
    # Change dimensionality of model, dynamics, transitions
    augmented_model.D = 2*model.D
    augmented_model.dynamics.D = 2*model.D
    augmented_model.emissions.D = 2*model.D
    # Initialize model parameters with correct dimension
    augmented_model.dynamics.As = np.zeros((model.K,2*model.D,2*model.D))
    augmented_model.dynamics.Vs = np.zeros((model.K,2*model.D, model.M))
    augmented_model.emissions.Cs = np.zeros((1,model.N,model.D*2))
    augmented_model.dynamics.Sigmas = np.ones((model.K,2*model.D,2*model.D))
    Sigmas = np.zeros((model.K,2*model.D,2*model.D))
    Sigmas_init = np.zeros((model.K,2*model.D,2*model.D))
    augmented_model.transitions.Rs = np.zeros((model.K,2*model.D))
    
    # Update As, Bs (Vs), Sigmas for each regime/discrete state
    for k in range(model.K):
        # A_aug = [[A, I],[0, I]]
        left_col = np.vstack((model.dynamics.As[k],np.zeros((model.D,model.D))))
        right_col = np.vstack(((np.eye(model.D),np.eye(model.D))))
        augmented_model.dynamics.As[k] = np.hstack((left_col,right_col))
        
        # B_aug = [[B],[0]]
        augmented_model.dynamics.Vs[k] = np.vstack((model.dynamics.Vs[k],np.zeros_like(model.dynamics.Vs[k])))
        
        # Q_aug = [[Q, 0], [0, Q_m]]
        if Q_m is None:
            Sigma_m = model.dynamics.Sigmas[k]
            Sigma_m_init = model.dynamics.Sigmas_init[k]
        left_col = np.vstack((model.dynamics.Sigmas[k],np.zeros((model.D,model.D))))
        right_col = np.vstack(((np.zeros((model.D,model.D)),Sigma_m)))
        # Make temporary Sigmas to overcome property setter
        Sigmas[k] = np.hstack((left_col,right_col))
        
        left_col = np.vstack((model.dynamics.Sigmas_init[k],np.zeros((model.D,model.D))))
        right_col = np.vstack(((np.zeros((model.D,model.D)),Sigma_m_init)))
        Sigmas_init[k] = np.hstack((left_col,right_col))
        
    # Use property setter
    augmented_model.dynamics.Sigmas = Sigmas
    augmented_model.dynamics.sigmasq_init = np.array([np.diag(S) for S in Sigmas_init])
    
    augmented_model.dynamics.mu_init = np.tile(model.dynamics.mu_init,(1,2))
    augmented_model.dynamics.bs = np.tile(model.dynamics.bs,(1,2))
    
    # Augment C matrix with zeros--aug dims dont contribute to observations
    augmented_model.emissions.Cs[0] = np.hstack((model.emissions.Cs[0],np.zeros_like(model.emissions.Cs[0])))
    
    # Augment R matrix with zeros--aug dims dont contribute to transitions
    augmented_model.transitions.Rs[:,:model.D] = model.transitions.Rs
    
    
    return augmented_model

def constrained_lqr(A, B, Q, R, x0, u0, Nlqr=50, Nsim=100,lowerlim=0,upperlim=10):
    """
    Following methods from Mare & Dona, 2007
    https://doi.org/10.1016/j.sysconle.2006.10.018
    
    Discrete-time constrained linear quadratic regulator for a linear system.
 
    Compute the optimal feedback gain given a linear system, cost matrices, constraints
    Compute the control variables that minimize the cumulative cost.
    Solve for P using the dynamic programming method.
 
    :param Q: The state cost matrix
    :param R: The input cost matrix
    :param N: number of timesteps for finite time horizon lqr
 
    :return: K: Optimal feedback gain K for the state to obtain u* = K(x_error) 

    """
    def Ahat_abc(a,b,c,L):
        prod = np.eye(A.shape[0])
        for j in range(a+1-b,a-c): # Upper lim in paper is a-1-c
            prod *= A - B @ L[j]  
        return prod
    
    def Lhat_r(r):
        resnum = np.zeros(G.shape)
        for i in range(1,r):
            resnum += (G - L[r-i]) @ Ahat_abc(r,i,0,L) @ B @ (G - L[r-i]) @ Ahat_abc(r,i,0,L) @ A
        resdenom = np.zeros((B.shape[1],B.shape[1]))
        for i in range(1,r):
            resdenom += ((G - L[r-i]) @ Ahat_abc(r,i,0,L) @ B)**2
        
        #return (G + resnum)/(1+resdenom)
        return np.linalg.lstsq(1+resdenom, G + resnum, rcond=None)[0]
    
    def hhat_r(r):
        resnum = np.zeros((B.shape[1],1))
        for i in range(1,r):
            resinnernum = np.zeros((A.shape[0],1))
            for p in range(1,i):
                resinnernum += Ahat_abc(r,i,p,L) @ B @ h[r-p]
            resnum += (G - L[r-i]) @ Ahat_abc(r,i,0,L) @ B @ ((G - L[r-i]) @ resinnernum + h[r-i])
        resdenom = np.zeros((B.shape[1],B.shape[1]))
        for i in range(1,r):
            resdenom += ((G - L[r-i]) @ Ahat_abc(r,i,0,L) @ B)**2
        
        return np.linalg.lstsq(-(1+resdenom), resnum, rcond=None)[0]
    
    # def L_r(r, x):
    #     if -Lhat[r] @ x[N-r] - hhat[r] > 0:
    #         return Lhat[r]
    #     else:
    #         return 0
    
    # def h_r(r, x):
    #     if -Lhat[r] @ x[N-r] - hhat[r] > 0:
    #         return hhat[r]
    #     else:
    #         return 0
    
    def constrain(L_t, h_t, xN_t,lowerlim=0,upperlim=10):
        out = np.squeeze(-L_t @ xN_t - h_t.T)
        Lout = copy.deepcopy(L_t)
        hout = copy.deepcopy(h_t)
        # For any row that results in a negative output, zero it instead
        Lout[out < lowerlim,:] = 0 
        hout[out < lowerlim] = lowerlim
        Lout[out > upperlim,:] = 0
        hout[out > upperlim] = -upperlim
        return Lout, hout
    
    
    # Solve ARE for terminal state weighting matrix
    P, diff = solveARE(A, B, Q, R, N=None) # Terminal weight P
    
    # Intermediate variables that will be used again later 
    Rbar = R + B.T @ P @ B
    G = np.linalg.inv(Rbar) @ B.T @ P @ A # This is K in the paper, notation change
    
    # Initialize with (Lhat[1], hhat[1]) = (K,0)  
    Lhat = [None] + [G]
    hhat = [None] + [np.zeros((B.shape[1],1))]
    L = copy.deepcopy(Lhat)
    h = copy.deepcopy(hhat)
    for r in range(2,Nlqr+1):
        # Intermediate terms
        Lhat.append(Lhat_r(r))
        hhat.append(hhat_r(r))
        
        Lr, hr = constrain(Lhat[-1],hhat[-1],x0[Nlqr-r,:],lowerlim=lowerlim,upperlim=upperlim)
        L.append(Lr)
        h.append(hr)
    
    Lhat.pop(0)
    hhat.pop(0)
    Lhat.reverse()
    hhat.reverse()
    Lhat = Lhat + [Lhat[-1]] * (Nsim-Nlqr)
    hhat = hhat + [hhat[-1]] * (Nsim-Nlqr)
    return np.array(Lhat), np.array(hhat)

#%% Extra
def lqr(actual_state_x, desired_state_xf, Q, R, A, B, dt, N=50):
    """
    Discrete-time linear quadratic regulator for a nonlinear system.
 
    Compute the optimal control inputs given a nonlinear system, cost matrices, 
    current state, and a final state.
     
    Compute the control variables that minimize the cumulative cost.
 
    Solve for P using the dynamic programming method.
 
    :param actual_state_x: The current state of the system 
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]
    :param desired_state_xf: The desired state of the system
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]   
    :param Q: The state cost matrix
        3x3 NumPy Array
    :param R: The input cost matrix
        2x2 NumPy Array
    :param dt: The size of the timestep in seconds -> float
 
    :return: u_star: Optimal action u for the current state 
        2x1 NumPy Array given the control input vector is
        [linear velocity of the car, angular velocity of the car]
        [meters per second, radians per second]
    """
    # We want the system to stabilize at desired_state_xf.
    x_error = actual_state_x - desired_state_xf
 
    # Solutions to discrete LQR problems are obtained using the dynamic 
    # programming method.
    # The optimal solution is obtained recursively, starting at the last 
    # timestep and working backwards.
    # You can play with this number
  
 
    # Create a list of N + 1 elements
    P = [None] * (N + 1)
     
    Qf = Q
 
    # LQR via Dynamic Programming
    P[N] = Qf
 
    # For i = N, ..., 1
    for i in range(N, 0, -1):
 
        # Discrete-time Algebraic Riccati equation to calculate the optimal 
        # state cost matrix
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
            R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)      
 
    # Create a list of N elements
    K = [None] * N
    u = [None] * N
 
    # For i = 0, ..., N - 1
    for i in range(N):
 
        # Calculate the optimal feedback gain K
        K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
 
        u[i] = K[i] @ x_error
 
    # Optimal control input is u_star
    u_star = u[N-1]
 
    return u_star, K, u

