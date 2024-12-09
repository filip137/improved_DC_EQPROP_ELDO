import numpy as np


#Initializers(
        ##weight_initialzier = Initializers(init_type="glorot", params={"L": 1e-7, "U": 8e-6, "g_max": 2e-5, "g_min": 1e-7})
        ## 

class Initializer:
    def __init__(self, init_type, params=None, L=1e-4, U=1e-1, g_min=None, g_max=None):
        self.init_type = init_type
        self.params = params if params is not None else {}
        self.L = L  # Lower bound
        self.U = U  # Upper bound
        self.g_min = g_min  # Minimum conductance
        self.g_max = g_max  # Maximum conductance

    def get_bounds(self):
        # Retrieve bounds from params or use class variables if not provided
        L = self.params.get("L", self.L)
        U = self.params.get("U", self.U)
        return L, U

    def clip_conductances(self, w):
        # Clip weights to be within the specified conductance range
        g_min = self.params.get("g_min", self.g_min)
        g_max = self.params.get("g_max", self.g_max)
        if g_min is not None:
            w = np.clip(w, g_min, None)
        if g_max is not None:
            w = np.clip(w, None, g_max)
        return w

    def initialize_weights(self, shape):
        # Initialize weights based on specified initialization type
        if self.init_type == 'random_uniform':
            return self.random_uniform(shape)
        elif self.init_type == 'glorot':
            return self.glorot(shape)

    def random_uniform(self, shape):
        # Generate weights using a uniform distribution within [L, U]
        L, U = self.get_bounds()
        return np.random.uniform(L, U, size=shape)

    def glorot(self, shape):
        # Generate weights using Glorot initialization, adjusted by bounds
        L, U = self.get_bounds()
        # Calculate fan sum and adjust upper bound accordingly
        fan_sum = np.sqrt(shape[0] + shape[1] + 1)
        return np.random.uniform(L, U / fan_sum, size=shape)
        # a = np.exp(np.random.uniform(np.log(L), np.log(U) / np.sqrt(shape[0] + shape[1] + 1), size=shape))
        # return a # log uniform
        

