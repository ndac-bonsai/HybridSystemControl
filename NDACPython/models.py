import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln

import ssm
from ssm import hmm, lds
from ssm.hmm import HMM
from ssm.lds import SLDS
from ssm.util import random_rotation, ensure_args_are_lists, softplus
from ssm.observations import Observations, AutoRegressiveDiagonalNoiseObservations
from ssm.transitions import Transitions, RecurrentTransitions, RecurrentOnlyTransitions
from ssm.init_state_distns import InitialStateDistribution
from ssm.emissions import _LinearEmissions, GaussianEmissions, PoissonEmissions
from ssm.preprocessing import factor_analysis_with_imputation, interpolate_data, pca_with_imputation
from ssm.optimizers import adam_step, rmsprop_step, sgd_step, lbfgs, bfgs, convex_combination
from ssm.primitives import hmm_normalizer

from ssmdm.misc import smooth

from ssmdm.accumulation import AccumulationRaceSoftTransitions, AccumulationRaceTransitions,\
    DDMTransitions,DDMCollapsingTransitions,DDMNonlinearCollapsingTransitions,DDMSoftTransitions,\
        EmbeddedAccumulationRaceTransitions
from ssmdm.accumulation import AccumulationObservations,AccumulationGLMObservations,\
    EmbeddedAccumulationObservations
from ssmdm.accumulation import AccumulationGaussianEmissions,AccumulationPoissonEmissions,RampStepPoissonEmissions
from ssmdm.accumulation import AccumulationInitialStateDistribution

import copy
import scipy

from tqdm import tqdm
from tqdm.auto import trange
from autograd.scipy.special import logsumexp
from autograd.tracer import getval
from autograd.misc import flatten
from autograd import value_and_grad



class Accumulation(HMM):
    def __init__(self, K, D, *, M,
                transitions="race",
                transition_kwargs=None,
                observations="acc",
                observation_kwargs=None,
                **kwargs):

        init_state_distn = AccumulationInitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log(np.concatenate(([0.999],(0.001/(K-1))*np.ones(K-1))))

        transition_classes = dict(
            racesoft=AccumulationRaceSoftTransitions,
            race=AccumulationRaceTransitions,
            ddmsoft=DDMSoftTransitions,
            ddm=DDMTransitions,
            ddmcollapsing=DDMCollapsingTransitions,
            ddmnlncollapsing=DDMNonlinearCollapsingTransitions)
        transition_kwargs = transition_kwargs or {}
        transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)

        observation_classes = dict(
            acc=AccumulationObservations,
            accglm=AccumulationGLMObservations)
        observation_kwargs = observation_kwargs or {}
        observation_distn = observation_classes[observations](K, D, M=M, **observation_kwargs)

        super().__init__(K, D, M=M,
                            init_state_distn=init_state_distn,
                            transitions=transitions,
                            observations=observation_distn)

class LatentAccumulation(SLDS):
    def __init__(self, N, K, D, *, M,
            transitions="race",
            transition_kwargs=None,
            dynamics="acc",
            dynamics_kwargs=None,
            emissions="gaussian",
            emission_kwargs=None,
            single_subspace=True,
            **kwargs):

        init_state_distn = AccumulationInitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log(np.concatenate(([0.9999],(0.0001/(K-1))*np.ones(K-1))))
        # init_state_distn.log_pi0 = np.log(np.concatenate(([1.0],(0.0/(K-1))*np.ones(K-1))))

        transition_classes = dict(
            racesoft=AccumulationRaceSoftTransitions,
            racesoft1d=AccumulationRaceSoft1DTransitions,
            race=AccumulationRaceTransitions,
            ddmsoft=DDMSoftTransitions,
            ddm=DDMTransitions,
            ddmcollapsing=DDMCollapsingTransitions,
            ddmnlncollapsing=DDMNonlinearCollapsingTransitions,
            embedded_race=EmbeddedAccumulationRaceTransitions,
            embedded_racesoft=EmbeddedAccumulationRaceSoftTransitions)
        self.transitions_label = transitions
        self.transition_kwargs = transition_kwargs
        transition_kwargs = transition_kwargs or {}
        transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)

        self.dynamics_kwargs = dynamics_kwargs
        dynamics_kwargs = dynamics_kwargs or {}
        dynamics_classes = dict(
            acc=AccumulationObservations,
            embedded_acc=EmbeddedAccumulationObservations,
            dawembedded=DAWEmbeddedAccumulationObservations,
            dawembeddedopto=DAWEmbeddedAccumulationObservationsWithOpto,
            acc1d=Accumulation1DObservations,
        )
        dynamics_kwargs = dynamics_kwargs or {}
        self.dynamics_kwargs = dynamics_kwargs
        dynamics = dynamics_classes[dynamics](K, D, M=M, **dynamics_kwargs)

        self.emissions_label = emissions
        emission_classes = dict(
            gaussian=AccumulationGaussianEmissions,
            poisson=AccumulationPoissonEmissions,
            rampstep=RampStepPoissonEmissions,
            dawpoisson=DAWAccumulationPoissonEmissions)#,
            # calcium=AccumulationCalciumEmissions)
        emission_kwargs = emission_kwargs or {}
        emissions = emission_classes[emissions](N, K, D, M=M,
            single_subspace=single_subspace, **emission_kwargs)

        super().__init__(N, K=K, D=D, M=M,
                            init_state_distn=init_state_distn,
                            transitions=transitions,
                            dynamics=dynamics,
                            emissions=emissions)

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None,
                   num_optimizer_iters=1000, num_em_iters=25,
                   betas=None, accum_log_sigmasq=None):

        # First initialize the observation model
        self.base_model = Accumulation(self.K, self.D, M=self.M,
                                       transitions=self.transitions_label,
                                       transition_kwargs=self.transition_kwargs,
                                       observation_kwargs=self.dynamics_kwargs)
        self.base_model.observations.Vs = self.dynamics.Vs
        self.base_model.observations.As = self.dynamics.As
        self.base_model.observations.Sigmas = self.dynamics.Sigmas
        self.emissions.initialize(self.base_model,
                                  datas, inputs, masks, tags)

        if self.emissions_label=="gaussian":
            # Get the initialized variational mean for the data
            xs = [self.emissions.invert(data, input, mask, tag)
                  for data, input, mask, tag in zip(datas, inputs, masks, tags)]
            xmasks = [np.ones_like(x, dtype=bool) for x in xs]

            # Now run a few iterations of EM on a ARHMM with the variational mean
            print("Initializing with an ARHMM using {} steps of EM.".format(num_em_iters))
            arhmm = hmm.HMM(self.K, self.D, M=self.M,
                            init_state_distn=copy.deepcopy(self.init_state_distn),
                            transitions=copy.deepcopy(self.transitions),
                            observations=copy.deepcopy(self.dynamics))

            arhmm.fit(xs, inputs=inputs, masks=xmasks, tags=tags,
                      method="em", num_em_iters=num_em_iters)

            self.init_state_distn = copy.deepcopy(arhmm.init_state_distn)
            self.transitions = copy.deepcopy(arhmm.transitions)
            self.dynamics = copy.deepcopy(arhmm.observations)

    @ensure_args_are_lists
    def monte_carlo_loglikelihood(self, datas, inputs=None, masks=None, tags=None, num_samples=100):
        """
        Estimate marginal likelihood p(y | theta) using samples from prior
        """
        trial_lls = []
        trial_sample_lls = []

        print("Estimating log-likelihood...")
        for data, input, mask, tag in zip(tqdm(datas), inputs, masks, tags):

            sample_lls = []
            samples = [self.sample(data.shape[0], input=input, tag=tag)
                       for sample in range(num_samples)]

            for sample in samples:

                z, x = sample[:2]
                sample_ll = np.sum(self.emissions.log_likelihoods(data, input, mask, tag, x))
                sample_lls.append(sample_ll)

                assert np.isfinite(sample_ll)

            trial_ll = logsumexp(sample_lls) - np.log(num_samples)
            trial_lls.append(trial_ll)
            trial_sample_lls.append(sample_lls)

        ll = np.sum(trial_lls)

        return ll, trial_lls, trial_sample_lls

class DAWEmbeddedAccumulationObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M, lags=1, D_emb=None, D_acc=None):
        super(DAWEmbeddedAccumulationObservations, self).__init__(K, D, M)

        assert D_emb + D_acc == D, "accum + emb dims must equal total dims"
        
        self.D_acc = D_acc
        self.D_emb = D_emb
        self.D = D_acc + D_emb
        self.M = M
        self.K = K

        # diagonal dynamics for each state
        # only learn accumulation dynamics for accumulation state
        self._a_diag = np.ones((D_acc,1))
        # Make random dynamics matrix for embedded dimensions
        
        # Diag = np.diag(np.random.uniform(low=0.7,high=0.8,size=(D_emb,)))# * \
        #                #(np.array([1 if x > 0.5  else -1 for x in np.random.random(Demb)])))
        
        # # Make random eigenvector matrix
        # Q = np.random.uniform(low=-1,high=1,size=(D_emb,D_emb)) 
        # self._a_emb = (Q @ Diag @ np.linalg.inv(Q)).flatten()
        #self._a_emb = random_rotation(D_emb)*np.random.uniform(low=0.7,high=0.8)
        
        # (M @ A)v = (M @ \lambda)v
        if D_emb == 1:
            self._a_emb = np.random.uniform(low=0.9,high=0.95,size=(D_emb,))*np.eye(D_emb) #@ random_rotation(D_emb)
        else:
            # a = np.random.uniform(low=0.9,high=0.95,size=(1,))
            # self._a_emb = scipy.linalg.block_diag(a,random_rotation(D_emb-1))
            self._a_emb = np.random.uniform(low=0.9,high=0.95,size=(D_emb,))*np.eye(D_emb) #@ random_rotation(D_emb)
        top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        Aaccstate = np.vstack((top_rows,bottom_rows))
        
        top_rows = np.hstack((np.eye(D_acc),np.zeros((D_acc,D_emb))))
        Adecstate = np.vstack((top_rows,bottom_rows))
        
        mask1 = np.vstack( (Aaccstate[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(Adecstate,(K-1,1,1)) ))
        self._As = mask1 + mask2
        
        # set input Accumulation params, one for each dimension
        # first D inputs are accumulated in different dimensions
        # rest of M-D inputs are applied to each dimension
        ### This is original, for first D_acc dims
        # self._betas = 0.1*np.ones(D_acc,)
        # r1 = self._betas*np.eye(D_acc, D_acc)
        # r2 = np.zeros((self.D_emb, M)) # is D_emb rows by input # of columns 
        # self.Vs[0] = np.vstack((r1, r2))
        ### This is new, for all dims
        self._betas = 0.1 * np.ones(M,)
        self._betas_emb = 0.1 * np.ones(D_emb*M,)
        r1 = self._betas*np.eye(D_acc, M)
        r2 = np.reshape(self._betas_emb,(D_emb,M))
        k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,M))))
        #k2 = np.tile(r2[None,:,:],(K,1,1))
        k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
        self.Vs = np.concatenate([k1,k2],axis=1)

        # self.Vs[0] = np.vstack((r1, r2))

        # for d in range(1,K):
        #     r1 = np.zeros((D_acc,M))
        #     r2 = self._betas_emb
        #     self.Vs[d] = np.vstack((r1, r2))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3)*np.ones(D,)
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self.bound_variance = 1e-4
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1,D))
        self.mu_init = np.vstack((acc_mu_init,np.ones((K-1,D))))

    @property
    def params(self):
        params = self._betas, self._betas_emb,  self.accum_log_sigmasq, self._a_diag, self._a_emb
        return params

    @params.setter
    def params(self, value):
        self._betas, self._betas_emb, self.accum_log_sigmasq, self._a_diag, self._a_emb = value

        K, D, M, D_acc, D_emb = self.K, self.D, self.M, self.D_acc, \
            self.D_emb

        # Update V
        ### This is original, for first D_acc dims
        # k1 = np.vstack((self._betas * np.eye(self.D_acc), np.zeros((self.D_emb,self.D_acc)))) # state K = 0
        # self.Vs = np.vstack((k1[None,:,:], np.zeros((K-1,D,M))))
        ### This is new, for all dims
        # k1 = self._betas * np.eye(self.D, self.M)
        # self.Vs = np.vstack((k1[None,:,:], np.zeros((K-1,D,M))))
        
        r1 = self._betas*np.eye(D_acc, M)
        k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,M))))
        r2 = np.reshape(self._betas_emb,(D_emb,M))
        #k2 = np.tile(r2[None,:,:],(K,1,1))
        k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
        self.Vs = np.concatenate([k1,k2],axis=1)
       
        # self._betas = 0.1 * np.ones(M,)
        # self._betas_emb = 0.1 * np.ones(D_emb*M,)
        # r1 = self._betas*np.eye(D_acc, M)
        # r2 = np.reshape(self._betas_emb,(D_emb,M))
        # k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,M))))
        # #k2 = np.tile(r2[None,:,:],(K,1,1))
        # k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
        # self.Vs = np.concatenate([k1,k2],axis=1)
        

        # for d in range(1,K):
        #     r1 = np.zeros((D_acc,M))
        #     r2 = self._betas_emb
        #     self.Vs[d] = np.vstack((r1, r2))

        # update sigmas
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # update A
        # if self.learn_A:
        # mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        # mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        # self._As = self._a_diag*mask1 + mask2
        
        # top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        # bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        # A_acc = np.vstack((top_rows,bottom_rows))
        # mask1 = np.vstack( (A_acc[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        # mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        # self._As = mask1 + mask2
        
        top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        #bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),np.reshape(self._a_emb,(D_emb,D_emb))))
        bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        Aaccstate = np.vstack((top_rows,bottom_rows))
        
        top_rows = np.hstack((np.eye(D_acc),np.zeros((D_acc,D_emb))))
        Adecstate = np.vstack((top_rows,bottom_rows))
        
        mask1 = np.vstack( (Aaccstate[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(Adecstate,(K-1,1,1)) ))
        self._As = mask1 + mask2

    # @property
    # def betas(self):
    #     return self._betas
    #
    # @betas.setter
    # def betas(self, value):
    #     assert value.shape == (self.D,)
    #     self._betas = value
    #     mask = np.vstack((np.eye(self.D)[None,:,:], np.zeros((self.K-1,self.D,self.D))))
    #     self.Vs = self._betas * mask

    def log_prior(self):
        alpha = 1.1 # or 0.02
        beta = 1e-3 # or 0.02
        dyn_vars = np.exp(self.accum_log_sigmasq)
        var_prior = np.sum( -(alpha+1) * np.log(dyn_vars) - np.divide(beta, dyn_vars))
        return var_prior

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, 
                continuous_expectations=None, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)


class DAWEmbeddedAccumulationObservationsWithOpto(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M, lags=1, D_emb=None, D_acc=None, Mphys=None, Mopto=None):
        super(DAWEmbeddedAccumulationObservationsWithOpto, self).__init__(K, D, M)

        assert D_emb + D_acc == D, "accum + emb dims must equal total dims"
        assert Mphys + Mopto == M, "accum + emb dims must equal total dims"
        
        self.D_acc = D_acc
        self.D_emb = D_emb
        self.D = D_acc + D_emb
        self.Mphys = Mphys
        self.Mopto = Mopto
        self.M = M
        self.K = K

        # diagonal dynamics for each state
        # only learn accumulation dynamics for accumulation state
        self._a_diag = np.ones((D_acc,1))
        # Make random dynamics matrix for embedded dimensions
        
        # Diag = np.diag(np.random.uniform(low=0.7,high=0.8,size=(D_emb,)))# * \
        #                #(np.array([1 if x > 0.5  else -1 for x in np.random.random(Demb)])))
        
        # # Make random eigenvector matrix
        # Q = np.random.uniform(low=-1,high=1,size=(D_emb,D_emb)) 
        # self._a_emb = (Q @ Diag @ np.linalg.inv(Q)).flatten()
        #self._a_emb = random_rotation(D_emb)*np.random.uniform(low=0.7,high=0.8)
        
        # (M @ A)v = (M @ \lambda)v
        self._a_emb = (np.random.uniform(low=0.7,high=0.9,size=(D_emb,))*np.eye(D_emb)) @ random_rotation(D_emb)
        
        top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        Aaccstate = np.vstack((top_rows,bottom_rows))
        
        top_rows = np.hstack((np.eye(D_acc),np.zeros((D_acc,D_emb))))
        Adecstate = np.vstack((top_rows,bottom_rows))
        
        mask1 = np.vstack( (Aaccstate[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(Adecstate,(K-1,1,1)) ))
        self._As = mask1 + mask2
        
        # set input Accumulation params, one for each dimension
        # first D inputs are accumulated in different dimensions
        # rest of M-D inputs are applied to each dimension
        ### This is original, for first D_acc dims
        # self._betas = 0.1*np.ones(D_acc,)
        # r1 = self._betas*np.eye(D_acc, D_acc)
        # r2 = np.zeros((self.D_emb, M)) # is D_emb rows by input # of columns 
        # self.Vs[0] = np.vstack((r1, r2))
        ### This is new, for all dims
        self._betas = 0.1 * np.ones(Mphys,)
        self._betas_emb = 0.1 * np.ones(D_emb*Mphys,)
        r1 = self._betas*np.eye(D_acc, Mphys)
        r2 = np.reshape(self._betas_emb,(D_emb,Mphys))
        k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,Mphys))))
        #k2 = np.tile(r2[None,:,:],(K,1,1))
        k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,Mphys))))
        betas_phys = np.concatenate([k1,k2],axis=1)
        self._betas_opto = 0.1 * np.ones((D, Mopto))
        k3 = np.tile(self._betas_opto[None,:,:], (K,1,1))
        
        self.Vs = np.concatenate([betas_phys,k3],axis=2)
        # self.Vs[0] = np.vstack((r1, r2))

        # for d in range(1,K):
        #     r1 = np.zeros((D_acc,M))
        #     r2 = self._betas_emb
        #     self.Vs[d] = np.vstack((r1, r2))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3)*np.ones(D,)
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self.bound_variance = 1e-4
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1,D))
        self.mu_init = np.vstack((acc_mu_init,np.ones((K-1,D))))

    @property
    def params(self):
        params = self._betas, self._betas_emb, self._betas_opto, self.accum_log_sigmasq, self._a_diag, self._a_emb
        return params

    @params.setter
    def params(self, value):
        self._betas, self._betas_emb, self._betas_opto, self.accum_log_sigmasq, self._a_diag, self._a_emb = value

        K, D, M, D_acc, D_emb, Mphys, Mopto = self.K, self.D, self.M, self.D_acc, \
            self.D_emb, self.Mphys, self.Mopto

        # Update V
        ### This is original, for first D_acc dims
        # k1 = np.vstack((self._betas * np.eye(self.D_acc), np.zeros((self.D_emb,self.D_acc)))) # state K = 0
        # self.Vs = np.vstack((k1[None,:,:], np.zeros((K-1,D,M))))
        ### This is new, for all dims
        # k1 = self._betas * np.eye(self.D, self.M)
        # self.Vs = np.vstack((k1[None,:,:], np.zeros((K-1,D,M))))

        # r1 = self._betas*np.eye(D_acc, M)
        # k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,M))))
        # r2 = np.reshape(self._betas_emb,(D_emb,M))
        # #k2 = np.tile(r2[None,:,:],(K,1,1))
        # k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,M))))
        # self.Vs = np.concatenate([k1,k2],axis=1)
        
        r1 = self._betas*np.eye(D_acc, Mphys)
        r2 = np.reshape(self._betas_emb,(D_emb,Mphys))
        k1 = np.vstack((r1[None,:,:], np.zeros((K-1,D_acc,Mphys))))
        #k2 = np.tile(r2[None,:,:],(K,1,1))
        k2 = np.vstack((r2[None,:,:], np.zeros((K-1,D_emb,Mphys))))
        betas_phys = np.concatenate([k1,k2],axis=1)     
        k3 = np.tile(self._betas_opto[None,:,:], (K,1,1))
        
        self.Vs = np.concatenate([betas_phys,k3],axis=2)
        

        # for d in range(1,K):
        #     r1 = np.zeros((D_acc,M))
        #     r2 = self._betas_emb
        #     self.Vs[d] = np.vstack((r1, r2))

        # update sigmas
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # update A
        # if self.learn_A:
        # mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        # mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        # self._As = self._a_diag*mask1 + mask2
        
        # top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        # bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        # A_acc = np.vstack((top_rows,bottom_rows))
        # mask1 = np.vstack( (A_acc[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        # mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        # self._As = mask1 + mask2
        
        top_rows = np.hstack((self._a_diag*np.eye(D_acc),np.zeros((D_acc,D_emb))))
        #bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),np.reshape(self._a_emb,(D_emb,D_emb))))
        bottom_rows = np.hstack((np.zeros((D_emb,D_acc)),self._a_emb))
        Aaccstate = np.vstack((top_rows,bottom_rows))
        
        top_rows = np.hstack((np.eye(D_acc),np.zeros((D_acc,D_emb))))
        Adecstate = np.vstack((top_rows,bottom_rows))
        
        mask1 = np.vstack( (Aaccstate[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(Adecstate,(K-1,1,1)) ))
        self._As = mask1 + mask2

    # @property
    # def betas(self):
    #     return self._betas
    #
    # @betas.setter
    # def betas(self, value):
    #     assert value.shape == (self.D,)
    #     self._betas = value
    #     mask = np.vstack((np.eye(self.D)[None,:,:], np.zeros((self.K-1,self.D,self.D))))
    #     self.Vs = self._betas * mask

    def log_prior(self):
        alpha = 1.1 # or 0.02
        beta = 1e-3 # or 0.02
        dyn_vars = np.exp(self.accum_log_sigmasq)
        var_prior = np.sum( -(alpha+1) * np.log(dyn_vars) - np.divide(beta, dyn_vars))
        return var_prior

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, 
                continuous_expectations=None, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)
        
class DAWAccumulationPoissonEmissions(PoissonEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="softplus", bin_size=1.0):
        super(DAWAccumulationPoissonEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace, link=link, bin_size=bin_size)
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0

    # Construct an emissions model
    @property
    def params(self):
        return self._Cs, self.ds

    @params.setter
    def params(self, value):
        self._Cs, self.ds = value

    def invert(self, data, input=None, mask=None, tag=None, clip=np.array([0.0,1.0]),augment=False):
#         yhat = self.link(np.clip(data, .1, np.inf))
        if self.bin_size < 1:
            yhat = smooth(data,20)
        else:
            yhat = smooth(data,5)
        # Only invert observable, non-augmented dimensions
        if augment:
            self.D = int(self.D/2)
            Cs = copy.deepcopy(self.Cs)
            self.Cs = self.Cs[:,:,:self.D]
        xhat = self.link(np.clip(yhat, 0.01, np.inf))
        xhat = self._invert(xhat, input=input, mask=mask, tag=tag)
        xhat = smooth(xhat,10)
        
        # Add back in augmented dimensions
        if augment:
            self.D = int(self.D*2)
            self.Cs = Cs
            xhat = np.hstack((xhat,np.zeros_like(xhat)))
        
        # DAW Commented
        # if self.bin_size < 1:
        #     xhat = np.clip(xhat, -0.95, 10)

        # in all models, x starts in between boundaries at [-1,1]
        if np.abs(xhat[0]).any()>1.0:
                xhat[0] = 0.05*npr.randn(1,self.D)
        return xhat
    
    # def log_likelihoods(self, data, input, mask, tag, x):
    #     assert data.dtype == int
    #     lambdas = self.mean(self.forward(np.clip(x,-15,15), input, tag))
    #     mask = np.ones_like(data, dtype=bool) if mask is None else mask
    #     lls = -gammaln(data[:,None,:] + 1) -lambdas + data[:,None,:] * np.log(lambdas)
    #     return np.sum(lls * mask[:, None, :], axis=2)
    
    def initialize(self, base_model, datas, inputs=None, masks=None, tags=None,
                   emission_optimizer="bfgs", num_optimizer_iters=1000):
        print("Initializing Emissions parameters...")

        if self.D == 1 and base_model.transitions.__class__.__name__ == "DDMTransitions":
        # if self.D == 0:
            d_init = np.mean([y[0:3] for y in datas],axis=(0,1))
            u_sum = np.array([np.sum(u) for u in inputs])
            y_end = np.array([y[-3:] for y in datas])
            u_l, u_u = np.percentile(u_sum, [20,80]) # use 20th and 80th percentile input
            y_U = y_end[np.where(u_sum>=u_u)]
            y_L = y_end[np.where(u_sum<=u_l)]
            C_init = (1.0/2.0)*np.mean((np.mean(y_U,axis=0) - np.mean(y_L,axis=0)),axis=0)
            self.Cs = C_init.reshape([1,self.N,self.D]) / self.bin_size
            self.ds = d_init.reshape([1,self.N]) / self.bin_size

        else:
            datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]

            Td = sum([data.shape[0] for data in datas])
            xs = [base_model.sample(T=data.shape[0],input=input)[1] for data, input in zip(datas, inputs)]
            def _objective(params, itr):
                self.params = params
                # self.Cs = params
                obj = 0
                obj += self.log_prior()
                for data, input, mask, tag, x in \
                    zip(datas, inputs, masks, tags, xs):
                    obj += np.sum(self.log_likelihoods(data, input, mask, tag, x))
                return -obj / Td

            # Optimize emissions log-likelihood
            optimizer = dict(bfgs=bfgs, lbfgs=lbfgs)[emission_optimizer]
            self.params = \
                optimizer(_objective,
                          self.params,
                          num_iters=num_optimizer_iters,
                          full_output=False)
                
class DAWAccumulationObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M, lags=1, learn_A=True, learn_V=False, learn_sig_init=False):
        super(DAWAccumulationObservations, self).__init__(K, D, M)

        # diagonal dynamics for each state
        # only learn dynamics for accumulation state
        self.learn_A = learn_A
        self._a_diag = np.ones((D,1))
        if self.learn_A:
            mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
            mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
            self._As = self._a_diag*mask1 + mask2
        else:
            self._As = np.tile(np.eye(D),(K,1,1))

        # set input Accumulation params, one for each dimension
        # first D inputs are accumulated in different dimensions
        # rest of M-D inputs are applied to each dimension
        self._betas = 0.1*np.ones(D,)
        self.learn_V = learn_V
        self._V = 0.1*np.ones((D, M-D)) # additional covariates, if they exist
        self.Vs[0] = np.hstack((self._betas*np.eye(D,D), self._V))
        for d in range(1,K):
            self.Vs[d] *= np.zeros((D,M))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3)*np.ones(D,)
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self.bound_variance = 1e-4
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self.learn_sig_init = learn_sig_init 
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = np.log(1e-3) * np.ones(D,)
            self._log_sigmasq_init = self.accum_log_sigmasq_init * mask1 + np.log(self.bound_variance) * mask2
        else:
            self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1,D))
        self.mu_init = np.vstack((acc_mu_init,np.ones((K-1,D))))

    @property
    def params(self):
        params = self._betas, self.accum_log_sigmasq
        params = params + (self._a_diag,) if self.learn_A else params
        params = params + (self._V,) if self.learn_V else params
        params = params + (self.accum_log_sigmasq_init) if self.learn_sig_init else params
        return params

    @params.setter
    def params(self, value):
        self._betas, self.accum_log_sigmasq = value[:2]
        if self.learn_A:
            self._a_diag = value[2]
        if self.learn_V:
            self._V = value[-1]
        #TODO fix above
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = value[-1]

        K, D, M = self.K, self.D, self.M

        # update V
        mask0 = np.hstack((np.eye(D), np.ones((D,M-D)))) # state K = 0
        mask01 = np.concatenate([np.zeros((2,D,D)), np.ones((2,D,M-D))],2)
        mask = np.vstack((mask0[None,:,:], mask01))

        # self.Vs = self._betas * mask
        self.Vs = np.hstack((np.diag(self._betas), self._V)) * mask

        # update sigmas
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        # self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = np.log(1e-3) * np.ones(D,)
            self._log_sigmasq_init = self.accum_log_sigmasq_init * mask1 + np.log(self.bound_variance) * mask2
        else:
            self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # update A
        # if self.learn_A:
        mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        self._As = self._a_diag*mask1 + mask2

class EmbeddedAccumulationRaceSoftTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=200, D_acc=None, D_emb=None):
        assert K == D_acc+1
        assert D_acc >= 1
        assert D_acc + D_emb == D
        super(EmbeddedAccumulationRaceSoftTransitions, self).__init__(K, D, M)

        # "Race" transitions with D+1 states
        # Transition to state d when x_d > 1.0
        # State 0 is the accumulation state
        # scale determines sharpness of the threshold
        # Transitions out of boundary states occur w/ very low probability
        #top_row = np.concatenate(([0.0],-scale*np.ones(D_acc)))
        #rest_rows = np.hstack((-scale*np.ones((D_acc,1)),-scale*np.ones((D_acc,D_acc)) + np.diag(2.0*scale*np.ones(D_acc))))
        #self.log_Ps = np.vstack((top_row,rest_rows))
        self.Ws = np.zeros((K,M))
        R1 = np.vstack((np.zeros(D_acc),scale*np.eye(D_acc)))
        R2 = np.zeros((K, D_emb))
        self.Rs = np.hstack((R1, R2))
        self.r = np.concatenate(([0],-scale*np.ones(D_acc)))

    @property
    def params(self):
        return ()

    @params.setter
    def params(self, value):
        pass
    

class Accumulation1DObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M, Mphys=2,lags=1, learn_A=True, learn_V=False, learn_betas=True, learn_sig_init=False):
        super(Accumulation1DObservations, self).__init__(K, D, M)
        D=1
        # diagonal dynamics for each state
        # only learn dynamics for accumulation state
        self.learn_A = learn_A
        self._a_diag = np.ones((D,1))
        if self.learn_A:
            mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
            mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
            self._As = self._a_diag*mask1 + mask2
        else:
            self._As = np.tile(np.eye(D),(K,1,1))

        # set input Accumulation params, each Mphys inputs are accumulated on one
        # dimension in opposite directions
        # rest of M-D inputs are applied to each dimension
        self.learn_betas = learn_betas
        self._betas = np.array([0.1,-0.1])*np.ones(Mphys,)
        self.learn_V = learn_V
        self._V = 0.1*np.ones((D, M-Mphys)) # additional covariates, if they exist
        self.Vs[0] = np.hstack((self._betas*np.eye(D,D), self._V))
        for d in range(1,K):
            if M > D+1:
                self.Vs[d] = np.hstack((np.zeros((D,M-(D+1))), self._V)) 
            else:
                self.Vs[d] *= np.zeros((D,M))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3)*np.ones(D,)
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self.bound_variance = 1e-4
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self.learn_sig_init = learn_sig_init 
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = np.log(1e-3) * np.ones(D,)
            self._log_sigmasq_init = self.accum_log_sigmasq_init * mask1 + np.log(self.bound_variance) * mask2
        else:
            self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1,D))
        self.mu_init = np.vstack((acc_mu_init,np.ones((K-1,D))))
        self.Mphys = Mphys
    
    @property
    def _learn_V(self):
        return self.learn_V

    @_learn_V.setter
    def _learn_V(self, value):
        self.learn_V = value
        
    @property
    def _learn_betas(self):
        return self.learn_betas

    @_learn_betas.setter
    def _learn_betas(self, value):
        assert type(value) == bool
        self.learn_betas = value
        
    @property
    def _learn_A(self):
        return self.learn_A

    @_learn_A.setter
    def _learn_A(self, value):
        assert type(value) == bool
        self.learn_A = value
        
    @property
    def params(self):
        params = self._betas, self.accum_log_sigmasq
        params = params + (self._a_diag,) if self.learn_A else params
        params = params + (self._V,) if self.learn_V else params
        params = params + (self.accum_log_sigmasq_init) if self.learn_sig_init else params
        return params

    @params.setter
    def params(self, value):
        if self.learn_betas:
            self._betas, self.accum_log_sigmasq = value[:2]
        if self.learn_A:
            self._a_diag = value[2]
        if self.learn_V:
            self._V = value[-1]
        #TODO fix above
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = value[-1]

        K, D, M = self.K, self.D, self.M

        # update V
        mask0 = np.ones((1,D,M)) # state K = 0
        mask01 = np.concatenate([np.zeros((K-1,D,self.Mphys)), np.ones((K-1,D,M-(D+1)))],2)
        mask = np.vstack((mask0, mask01))

        # self.Vs = self._betas * mask
        self.Vs = np.hstack((self._betas[None,:], self._V)) * mask

        # update sigmas
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        # self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2
        if self.learn_sig_init:
            self.accum_log_sigmasq_init = np.log(1e-3) * np.ones(D,)
            self._log_sigmasq_init = self.accum_log_sigmasq_init * mask1 + np.log(self.bound_variance) * mask2
        else:
            self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # update A
        # if self.learn_A:
        mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        self._As = self._a_diag*mask1 + mask2

    # @property
    # def betas(self):
    #     return self._betas
    #
    # @betas.setter
    # def betas(self, value):
    #     assert value.shape == (self.D,)
    #     self._betas = value
    #     mask = np.vstack((np.eye(self.D)[None,:,:], np.zeros((self.K-1,self.D,self.D))))
    #     self.Vs = self._betas * mask

    def log_prior(self):
        alpha = 1.1 # or 0.02
        beta = 1e-3 # or 0.02
        dyn_vars = np.exp(self.accum_log_sigmasq)
        var_prior = np.sum( -(alpha+1) * np.log(dyn_vars) - np.divide(beta, dyn_vars))
        return var_prior

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, 
                continuous_expectations=None, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

class AccumulationRaceSoft1DTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=100):
        assert K == D+2
        assert D >= 1
        super(AccumulationRaceSoft1DTransitions, self).__init__(K, D, M)

        # Like Race Transitions but soft boundaries
        # Transitions depend on previous x only
        # Transition to state d when x_d > 1.0
        self.Ws = np.zeros((K,M))
        self.Rs = np.vstack((np.zeros(D),scale*np.eye(D),-scale*np.eye(D)))
        self.r = np.concatenate(([0],-scale*np.ones(D),-scale*np.ones(D)))

    @property
    def params(self):
        return ()

    @params.setter
    def params(self, value):
        pass

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        pass