from .layer_class import *
from initializer import *
from .simulation_parameters import SimulationParameters

def initialize_network_layers(simulation_parameters):
    # Unpack initializer configuration correctly
    init_config = simulation_parameters.initializer["initializer"]
    iparams = init_config["init_type"]
    wparams = init_config["params"]
    seed = init_config["seed"]
    weight_initializer = Initializer(iparams, wparams, seed)
    print(f"Doing simulations for {wparams}")
    
    # Extract layer parameters
    layer_params = simulation_parameters.layer_parameters
    simulation_type = layer_params["simulation_type"]
    fc_layers       = layer_params["network_size"]
    freq            = layer_params["freq"]
    neuron          = layer_params["neuron"]
    nudging_mode    = layer_params["nudging_mode"]
    bounds          = layer_params["bounds"]
    synapse         = layer_params["synapse"] 
    gamma1, gamma2  = layer_params["gamma_values"]
    cs_bias         = layer_params["cs_bias"]
    non_lin         = layer_params["non_lin"]
    
    
    transient_params = simulation_parameters.layer_parameters
    
    
    
    layers = []
    
    # Input Layer
    input_layer = InputLayer(
        n_of_nodes=fc_layers[0],
        freq=freq,
        simulation_type=simulation_type,
        which_layer=0,
    )
    layers.append(input_layer)
    
    # First Dense Layer
    layer1 = DenseLayer(
        n_of_inputs=fc_layers[0],
        n_of_outputs=fc_layers[1],
        synapse=synapse,
        bounds=bounds,
        gamma=gamma1,
        initializer=weight_initializer,
        freq=freq,
        simulation_type=simulation_type,
        which_layer=0,
    )
    layer1.initialize_W()
    #layer1.update_synapse_dict()
    layers.append(layer1)
    
    # Non-linear Layer
    layer2 = NonLinearLayer(
        n_of_nodes=fc_layers[1],
        neuron_type=neuron,
        cs_bias=cs_bias,
        non_lin=non_lin,
        which_layer=0,
        freq=freq,
        simulation_type=simulation_type,
    )
    layers.append(layer2)
    
    # Second Dense Layer
    layer3 = DenseLayer(
        n_of_inputs=fc_layers[1],
        n_of_outputs=fc_layers[2],
        synapse=synapse,
        bounds=bounds,
        gamma=gamma2,
        initializer=weight_initializer,
        freq=freq,
        simulation_type=simulation_type,
        which_layer=1,
    )
    layer3.initialize_W()
    #layer3.update_synapse_dict()
    layers.append(layer3)
    
    # Output Layer
    layer4 = OutputLayer(
        n_of_nodes=fc_layers[2],
        freq=freq,
        simulation_type=simulation_type,
        nudging_mode=nudging_mode,
        cs_bias=cs_bias,
        which_layer=1,
    )
    layers.append(layer4)
    
    return layers

