"""
28/6
netlist_generation

A package for building and simulating SPICE netlists.
"""

# 1) Import the classes/functions you want to expose at the package level
from .layer_class           import BaseLayer, InputLayer, DenseLayer, NonLinearLayer, OutputLayer
from .netlist_generation    import netlist_builder
from .layers_initialization import initialize_network_layers
from .simulation_parameters import SimulationParameters
from .layers_initialization import initialize_network_layers

# 2) Define __all__ so "from netlist_generation import *" only pulls these
__all__ = [
    "BaseLayer",
    "InputLayer",
    "DenseLayer",
    "NonLinearLayer",
    "OutputLayer",
    "netlist_builder",
    "SimulationParameters",
    "initialize_network_layers"
]
