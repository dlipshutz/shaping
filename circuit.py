# Title: circuit.py
# Description: supplementary code for NeurIPS 2024 submission "Shaping the distribution of neural responses with interneurons in a recurrent circuit model"
# Author: anonymous

##############################

# Imports

import numpy as np

from scipy import optimize
from scipy.special import gamma
from scipy.special import digamma
from scipy.special import polygamma
from scipy.special import erf
from scipy.stats import ortho_group

##############################

# useful functions

def moment(p): return np.sqrt(2**p/np.pi)*gamma((p+1)/2)
def moment_prime(p): return (1/2)*np.sqrt(2**p/np.pi)*gamma((p+1)/2)*(np.log(2) + digamma((p+1)/2))
def a_function(theta): return np.exp(np.maximum(2*theta-3.85,0)**1.95)
def b_function(theta): return np.exp(np.maximum(theta**2.32-6,0))
def a_prime(theta): return a_function(theta)*3.9*(np.maximum(2*theta-3.85,0)**0.95)
def b_prime(theta): return b_function(theta)*2.32*(theta**1.32)
def linear_constraint_function(z): return (z**2 - 1)/2
def monomial_constraint_function(theta, z): return (np.abs(z)**(theta+1) - moment(theta+1))/(theta+1)
def monomial_theta_update(theta, z): 
    if abs(z)>0: return np.log(np.abs(z))*np.abs(z)**(theta+1) - moment_prime(theta+1)
    else: return 0
def constraint_function(theta, z): return a_function(theta)*linear_constraint_function(z) + b_function(theta)*monomial_constraint_function(theta, z)
def constraint_function_theta_update(theta, z): return a_prime(theta)*linear_constraint_function(z) + b_prime(theta)*monomial_constraint_function(theta, z) - b_function(theta)*monomial_constraint_function(theta,z)/(theta+1) + b_function(theta)*monomial_theta_update(theta, z)/(theta+1)

class circuit:
    """
    Parameters:
    ====================
    s_dim           -- Dimension of signals
    n_dim           -- Dimension of interneurons
    g0              -- Initialization of gains, size n_dim
    theta0          -- Initialization of theta parameters, size n_dim
    W0              -- Initialization of weight matrix W, size s_dim by n_dim
    g_hist          -- History of g paramters
    theta_hist      -- History of theta parameters
    W_hist          -- History of weights
    dg_hist         -- History of dg paramters
    dtheta_hist     -- History of dtheta parameters
    dW_hist         -- History of weights
    
    Methods:
    ========
    interneuron_output()
    response()
    fit_next()
    """

    def __init__(self, s_dim, n_dim, dataset=None, g0=None, theta0=None, W0=None):

        if W0 is not None:
            assert W0.shape == (s_dim,n_dim), "The shape of W0 must be (s_dim,n_dim)=(%d,%d)" % (s_dim, n_dim)
            W = W0
        else:
            if s_dim==1: W = np.ones((s_dim,n_dim))
            else: W = ortho_group.rvs(s_dim)@np.eye(s_dim,n_dim)@ortho_group.rvs(n_dim)
                        
        if g0 is not None: g = g0
        else: g = np.ones((n_dim))
        
        if theta0 is not None: theta = theta0
        else: theta = 2*np.ones((n_dim))
            
        self.s_dim = s_dim
        self.n_dim = n_dim
        
        self.g = g
        self.theta = theta
        self.W = W
        
        self.g_hist = np.array([g])
        self.theta_hist = np.array([theta])
        self.W_hist = np.array([W.flatten()])
        
        self.dg_hist = []
        self.dtheta_hist = []
        self.dW_hist = []
        
    def interneuron_output(self, response):
        
        n_dim, W, g, theta = self.n_dim, self.W, self.g, self.theta

        n = np.zeros(n_dim)

        z = W.T@response

        for i in range(n_dim): n[i] = g[i] * (a_function(theta[i])*z[i] + b_function(theta[i])*np.sign(z[i])*np.abs(z[i])**theta[i])

        return n

    def response(self, stimuli):
        
        # def func(r): return stimuli - r - self.W@self.interneuron_output(r)
        def func(r): return stimuli - self.W@self.interneuron_output(r)

        r = optimize.root(func, stimuli)

        return r.x

    def fit_next(self, batch, lr_g=1e-2, lr_theta=1e-2, lr_w=1e-3, report_responses=False):
        
        s_dim, n_dim, W, g, theta  = self.s_dim, self.n_dim, self.W, self.g, self.theta

        batch_size = batch.shape[1]
        
        R = np.zeros((s_dim, batch_size))
        N = np.zeros((n_dim, batch_size))
        dg = np.zeros((n_dim, batch_size))
        dtheta = np.zeros((n_dim, batch_size))

        # run iterations
                
        for t in range(batch_size):

            # neural activities
            
            R[:,t] = self.response(batch[:,t])
            N[:,t] = self.interneuron_output(R[:,t])
                                    
            z = W.T@R[:,t]
            
            for i in range(n_dim): 
                
                dg[i, t] = g[i]*constraint_function(theta[i], z[i]) # we scale the update dg by g to avoid large updates when g is small (this corresponds to an exact gradient step when g is replaced with exp(g) in the objective)
                dtheta[i, t] = constraint_function_theta_update(theta[i], z[i])
 
        # gain updates
    
        dg = dg.mean(axis=1)
        g = np.maximum(g + lr_g * dg, 0)
        
        # activation function updates

        dtheta = dtheta.mean(axis=1)
        theta = np.minimum(np.maximum(theta + lr_theta * dtheta, 1), 3) # added bounds to theta to prevent numerical instability

        # Hebbian weight updates
        
        dW = R@N.T/batch_size
        W = W + lr_w * dW
        
        # normalize weights
        
        for i in range(n_dim): W[:,i] = W[:,i]/np.linalg.norm(W[:,i])

        # save parameters
        
        self.g = g
        self.theta = theta
        self.W = W
        
        self.g_hist = np.append(self.g_hist, np.array([g]), axis=0)
        self.theta_hist = np.append(self.theta_hist, np.array([theta]), axis=0)
        self.W_hist = np.append(self.W_hist, np.array([W.flatten()]), axis=0)
        
        self.dg_hist.append(dg)
        self.dtheta_hist.append(dtheta)
        self.dW_hist.append(dW.flatten())
        
        if report_responses==True:
            return R