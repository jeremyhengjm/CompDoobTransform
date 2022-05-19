"""
A module to approximate functions with neural networks.
"""

import torch
import torch.nn.functional as F
from torch import nn

class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final = False, activation_fn = torch.nn.LeakyReLU):
        """
        Parameters
        ----------    
        input_dim : int specifying input_dim of input 

        layer_widths : list specifying width of each layer 
            (len is the number of layers, and last element is the output input_dim)

        activate_final : bool specifying if activation function is applied in the final layer

        activation_fn : activation function for each layer        
        """
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            # # same init for everyone
            # torch.nn.init.constant_(layers[-1].weight, 0)
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x

class V0_Network(torch.nn.Module):

    def __init__(self, dimension_state, dimension_obs, config):
        """
        Parameters
        ----------
        dimension_state : int specifying state dimension
        dimension_obs : int specifying observation dimension
        config : dict containing      
            layers : list specifying width of each layer 
        """
        super().__init__()
        input_dimension = dimension_state + dimension_obs        
        layers = config['layers']
        self.standardization = config['standardization']
        self.net = MLP(input_dimension, 
                       layer_widths = layers + [1],
                       activate_final = False,
                       activation_fn = torch.nn.LeakyReLU())

    def forward(self, x, y):
        """
        Parameters
        ----------
        x : state (N, d)
        
        y : observation (1, p) or (N, p)
                        
        Returns
        -------    
        out :  output (N)
        """
        
        N = x.shape[0]
        if len(y.shape) == 1:
            y_ = y.repeat((N, 1))
        else:
            y_ = y            

        x_c = (x - self.standardization['x_mean']) / self.standardization['x_std']
        y_c = (y_ - self.standardization['y_mean']) / self.standardization['y_std']            
        h = torch.cat([x_c, y_c], -1) # size (N, d+p)            
        out = torch.squeeze(self.net(h)) # size (N)
            
        return out

class Z_Network(torch.nn.Module):

    def __init__(self, dimension_state, dimension_obs, config):
        """
        Parameters
        ----------
        dimension_state : int specifying state dimension
        dimension_obs : int specifying observation dimension
        config : dict containing      
            layers : list specifying width of each layer
        """
        super().__init__()
        input_dimension = dimension_state + dimension_obs + 1
        layers = config['layers']
        self.standardization = config['standardization']
        self.net = MLP(input_dimension, 
                       layer_widths = layers + [dimension_state],
                       activate_final = False,
                       activation_fn = torch.nn.LeakyReLU())

    def forward(self, t, x, y):
        """
        Parameters
        ----------
        t : time step (N, 1)

        x : state (N, d)

        y : observation (1, p) or (N, p)
                        
        Returns
        -------    
        out :  output (N, d)
        """
        
        N = x.shape[0]
        if len(t.shape) == 0:
            t_ = t.repeat((N, 1))
        else:
            t_ = t
        
        if len(y.shape) == 1:
            y_ = y.repeat((N, 1))
        else: 
            y_ = y

        x_c = (x - self.standardization['x_mean']) / self.standardization['x_std']
        y_c = (y_ - self.standardization['y_mean']) / self.standardization['y_std']            
        h = torch.cat([t_, x_c, y_c], -1) # size (N, 1+d+p)        
        out = self.net(h) # size (N, d)
        return out
