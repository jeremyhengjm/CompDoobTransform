"""
A module to implement computational Doob's h-transforms.
"""

import torch    
import torch.nn.functional as F
from CompDoobTransform.neuralnet import V0_Network, Z_Network
from CompDoobTransform.utils import resampling

def construct_time_discretization(T, M):    
    time = torch.linspace(0.0, T, M + 1, device = T.device)
    stepsizes = (T / M) * torch.ones(M, device = T.device)
    return (time, stepsizes)

class model(torch.nn.Module):
    
    def __init__(self, state, obs, num_steps, net_config, device = 'cpu'):
        """
        Inputs
        ----------
        state : dict for objects relating to latent state process
        obs : dict for objects relating to observations
        num_steps : number of time-discretization steps M
        net_config : configuration of neural network for V0 and Z process
        device : device for computation
        """
        super().__init__()
        
        # latent state process
        self.d = state['dim']
        self.b = state['drift']
        self.sigma = state['sigma']
        self.T = state['terminal_time']
        self.initial = state['initial']

        # observations
        self.p = obs['dim']
        self.obs_log_density = obs['log_density']
        self.observation = obs['observation']

        # time discretization
        self.M = num_steps
        (self.time, self.stepsizes) = construct_time_discretization(self.T, num_steps)

        # initialize approximation of neural network for V0 and Z
        self.V0_net = V0_Network(self.d, self.p, net_config['V0'])
        self.V0_net.to(device)
        self.Z_net = Z_Network(self.d, self.p, net_config['Z'])
        self.Z_net.to(device)
        self.training_parameters = [{'params': self.V0_net.parameters()}, {'params': self.Z_net.parameters()}]
        
        # device for computation
        self.device = device

    def simulate_diffusion(self, initial_states):
        """
        Simulate stochastic differential equation for X using Euler-Maruyama discretization.

        Parameters
        ----------
        initial_states : initial states of X process (N, d)
                        
        Returns
        -------    
        X : X process at terminal time (N, d)
        """

        # initialize and preallocate
        N = initial_states.shape[0]
        X = initial_states # size (N, d)
        M = self.M
        d = self.d
        
        # simulate X process forwards in time
        for m in range(M):
            stepsize = self.stepsizes[m]
            drift = self.b(X)
            euler = X + stepsize * drift
            W = torch.sqrt(stepsize) * torch.randn(X.shape, device = self.device) # size (N, d)
            X = euler + self.sigma * W

        return X

    def simulate_SDEs(self, initial_states, initial_values, observations):
        """
        Simulate stochastic differential equation for X and V using Euler-Maruyama discretization.

        Parameters
        ----------
        initial_states : initial states of X process (N, d)

        initial_values : initial values of V process (N)

        observations : observations Y at terminal time (N, p)
                        
        Returns
        -------
        tuple containing
            X : X process at terminal time (N, d)
            V : V process at terminal time (N)
        """

        # initialize and preallocate
        N = initial_states.shape[0]
        X = initial_states # size (N, d)
        V = initial_values # size (N)   
        Y = observations # size (N, p)     
        M = self.M
        d = self.d
        
        for m in range(M):
            # time step
            stepsize = self.stepsizes[m]
            t = self.time[m]

            # Brownian increment
            W = torch.sqrt(stepsize) * torch.randn(X.shape, device = self.device) # size (N, d)

            # simulate V process forwards in time            
            Z = self.Z_net(t, X, Y) # size (N, d)
            drift_V = 0.5 * torch.sum(torch.square(Z), 1) # size (N)
            euler_V = V + stepsize * drift_V # size (N)
            V = euler_V + torch.sum(Z * W, 1) # size (N)

            # simulate X process forwards in time            
            drift_X = self.b(X)
            euler_X = X + stepsize * drift_X
            X = euler_X + self.sigma * W            

        return X, V

    def simulate_controlled_SDEs(self, initial_states, initial_values, observations):
        """
        Simulate controlled stochastic differential equations for X and V using Euler-Maruyama discretization.

        Parameters
        ----------    
        initial_states : initial states of X process (N, d)

        initial_values : initial values of V process (N)

        observations : observations Y at terminal time (N, p)
                        
        Returns
        -------
        tuple containing
            X : X process at terminal time (N, d)
            V : V process at terminal time (N)
        """

        # initialize and preallocate
        N = initial_states.shape[0]
        X = initial_states # size (N, d)
        V = initial_values # size (N)   
        Y = observations # size (N, p)     
        M = self.M
        d = self.d
        
        for m in range(M):
            # time step
            stepsize = self.stepsizes[m]
            t = self.time[m]

            # Brownian increment
            W = torch.sqrt(stepsize) * torch.randn(X.shape, device = self.device) # size (N, d)

            # simulate V process forwards in time
            Z = self.Z_net(t, X, Y) # size (N, d)
            control = - Z.clone().detach()
            drift_V = 0.5 * torch.sum(torch.square(Z), 1) + torch.sum(control * Z, 1) # size (N)
            euler_V = V + stepsize * drift_V # size (N)
            V = euler_V + torch.sum(Z * W, 1) # size (N)

            # simulate X process forwards in time
            drift_X = self.b(X) + self.sigma * control
            euler_X = X + stepsize * drift_X
            X = euler_X + self.sigma * W

        return X, V

    def train_standard(self, optim_config):
        """
        Train approximations using deep backward stochastic differential framework.

        Parameters
        ----------  
        optim_config : configuration of optimizer   
                        
        Returns
        -------   
        loss : value of loss function during learning (num_iterations)
        """

        # optimization configuration
        minibatch = optim_config['minibatch']
        num_obs_per_batch = optim_config['num_obs_per_batch']
        N = minibatch * num_obs_per_batch
        num_iterations = optim_config['num_iterations']
        learning_rate = optim_config['learning_rate']
        optimizer = torch.optim.Adam(self.training_parameters, lr = learning_rate)

        # optimization
        loss_values = torch.zeros(num_iterations, device = self.device)
        for i in range(num_iterations): 
            # simulate initial states X0
            X0 = self.initial(N)

            # simulate Y observations
            Y = self.observation(num_obs_per_batch).repeat((minibatch,1)) # size (N, 1)

            # evaluate initial values V0
            V0 = self.V0_net(X0, Y) # size (N)

            # simulate X and V processes 
            X, V = self.simulate_SDEs(X0, V0, Y)

            # loss function
            loss = F.mse_loss(V, -self.obs_log_density(X, Y))
                
            # backpropagation
            loss.backward()
    
            # optimization step and zero gradient
            optimizer.step()
            optimizer.zero_grad()

            # store loss 
            current_loss = loss.item()
            loss_values[i] = current_loss
            if (i == 0) or ((i+1) % 50 == 0):
                print('Optimization iteration:', i+1, 'Loss:', current_loss)
                
         # output loss values
        self.loss = loss_values

    def train_iterative(self, optim_config):
        """
        Train approximations using iterative stages and deep backward stochastic differential framework.

        Parameters
        ----------
        optim_config : configuration of optimizer
                        
        Returns
        -------    
        loss : value of loss function during learning (num_iterations)
        """

        # optimization configuration
        minibatch = optim_config['minibatch']
        num_obs_per_batch = optim_config['num_obs_per_batch']
        N = minibatch * num_obs_per_batch
        num_iterations = optim_config['num_iterations']
        learning_rate = optim_config['learning_rate']
        initial_required = optim_config['initial_required']
        optimizer = torch.optim.Adam(self.training_parameters, lr = learning_rate)

        # optimization
        loss_values = torch.zeros(num_iterations, device = self.device)

        for i in range(num_iterations):
            # simulate initial states X0
            X0 = self.initial(N)

            # simulate Y observations
            Y = self.observation(num_obs_per_batch).repeat((minibatch,1)) # size (N, 1)

            # evaluate initial values V0
            V0 = self.V0_net(X0, Y) # size (N)            

            if i == 0 and initial_required: # first training stage
                # simulate X and V processes 
                X, V = self.simulate_SDEs(X0, V0, Y)

            else: # subsequent iterative stages
                # simulate controlled X and V processes 
                X, V = self.simulate_controlled_SDEs(X0, V0, Y)

            # loss function
            loss = F.mse_loss(V, -self.obs_log_density(X, Y))   
            
            # backpropagation
            loss.backward()
 
            # optimization step and zero gradient
            optimizer.step()
            optimizer.zero_grad()

            # store loss 
            current_loss = loss.item()
            loss_values[i] = current_loss
            if (i == 0) or ((i+1) % 50 == 0):
                print('Optimization iteration:', i+1, 'Loss:', current_loss)
        
        # output loss values
        self.loss = loss_values

    def run_BPF(self, initial_states, observations, num_samples, full_path = False):
        """
        Run bootstrap particle filter.
        
        Parameters
        ----------
        initial_states : initial states of X process (N, d)
        
        observations : sequence of observations to be filtered (K, p)

        num_samples : sample size (int)

        full_path : if full path of X is required (bool)

        Returns
        -------
        dict containing:    
            states : X process at observation times (N, K+1, d) or (N, K*M+1, d) if full_path == True
            ess : effective sample sizes at unit times (K+1)        
            log_norm_const : log-normalizing constant estimates (K+1)
        """
        
        # initialize and preallocate
        N = num_samples
        Y = observations
        K = observations.shape[0]        
        d = self.d
        M = self.M
        X = initial_states        
        if full_path:
            states = torch.zeros(N, K*M+1, d, device = self.device)
        else:
            states = torch.zeros(N, K+1, d, device = self.device)
        states[:, 0, :] = X
        ess = torch.zeros(K+1, device = self.device)
        ess[0] = N
        log_norm_const = torch.zeros(K+1, device = self.device)
        log_ratio_norm_const = torch.tensor(0.0, device = self.device)
        
        # each observation
        for k in range(K):        
            
            # each time interval
            for m in range(M):
                # time step 
                stepsize = self.stepsizes[m]
                s = self.time[m]

                # Brownian increment
                W = torch.sqrt(stepsize) * torch.randn(N, d, device = self.device) # size (N, d)

                # simulate X process forwards in time
                euler_X = X + stepsize * self.b(X)
                X = euler_X + self.sigma * W
                if full_path:
                    index = k*M + m + 1
                    states[:, index, :] = X

            # compute and normalize weights, compute ESS and normalizing constant
            log_weights = self.obs_log_density(X, Y[k,:])
            max_log_weights = torch.max(log_weights)
            weights = torch.exp(log_weights - max_log_weights)
            normalized_weights = weights / torch.sum(weights)
            ess[k+1] = 1.0 / torch.sum(normalized_weights**2)
            log_ratio_norm_const = log_ratio_norm_const + torch.log(torch.mean(weights)) + max_log_weights
            log_norm_const[k+1] = log_ratio_norm_const

            # resampling            
            ancestors = resampling(normalized_weights, N)
            X = X[ancestors,:]

            # store states 
            if full_path:
                index_start = k*M + 1
                index_end = k*M + M + 1
                states[:, index_start:index_end, :] = states[ancestors, index_start:index_end, :]
            else:
                states[:, k+1, :] = X

        # output
        output = {'states' : states, 'ess' : ess, 'log_norm_const' : log_norm_const}

        return output

    def run_APF(self, initial_states, observations, num_samples, full_path = False):
        """
        Run auxiliary particle filter.
        
        Parameters
        ----------
        initial_states : initial states of X process (N, d)
        
        observations : sequence of observations to be filtered (K, p)

        num_samples : sample size (int)

        full_path : if full path of X is required (bool)

        Returns
        -------
        dict containing:    
            states : X process at observation times (N, K+1, d) or (N, K*M+1, d) if full_path == True
            ess : effective sample sizes at unit times (K+1)        
            log_norm_const : log-normalizing constant estimates (K+1)
        """
        
        # initialize and preallocate
        N = num_samples
        Y = observations
        K = observations.shape[0]        
        d = self.d
        M = self.M
        X = initial_states        
        if full_path:
            states = torch.zeros(N, K*M+1, d, device = self.device)
        else:
            states = torch.zeros(N, K+1, d, device = self.device)
        states[:, 0, :] = X
        ess = torch.zeros(K+1, device = self.device)
        ess[0] = N
        log_norm_const = torch.zeros(K+1, device = self.device)
        log_ratio_norm_const = torch.tensor(0.0, device = self.device)
        
        # each observation
        for k in range(K):        

            # evaluate initial values V0
            with torch.no_grad():
                V0 = self.V0_net(X, Y[k,:]) # size (N)            
            V = V0.clone()
            
            # each time interval
            for m in range(M):
                # time step 
                stepsize = self.stepsizes[m]
                t = self.time[m]

                # Brownian increment
                W = torch.sqrt(stepsize) * torch.randn(N, d, device = self.device) # size (N, d)
                
                # simulate V process forwards in time
                with torch.no_grad():
                    Z = self.Z_net(t, X, Y[k,:]) # size (N, d)
                control = - Z.clone()
                drift_V = - 0.5 * torch.sum(torch.square(Z), 1) # size (N)                
                euler_V = V + stepsize * drift_V # size (N)
                V = euler_V + torch.sum(Z * W, 1) # size (N)

                # simulate X process forwards in time
                drift_X = self.b(X) + self.sigma * control
                euler_X = X + stepsize * drift_X
                X = euler_X + self.sigma * W
                if full_path:
                    index = k*M + m + 1
                    states[:, index, :] = X

            # compute and normalize weights, compute ESS and normalizing constant
            log_weights = V + self.obs_log_density(X, Y[k,:]) - V0
            max_log_weights = torch.max(log_weights)
            weights = torch.exp(log_weights - max_log_weights)
            normalized_weights = weights / torch.sum(weights)
            ess[k+1] = 1.0 / torch.sum(normalized_weights**2)
            log_ratio_norm_const = log_ratio_norm_const + torch.log(torch.mean(weights)) + max_log_weights
            log_norm_const[k+1] = log_ratio_norm_const

            # resampling            
            ancestors = resampling(normalized_weights, N)
            X = X[ancestors,:]

            # store states 
            if full_path:
                index_start = k*M + 1
                index_end = k*M + M + 1
                states[:, index_start:index_end, :] = states[ancestors, index_start:index_end, :]
            else:
                states[:, k+1, :] = X

        # output
        output = {'states' : states, 'ess' : ess, 'log_norm_const' : log_norm_const}

        return output
