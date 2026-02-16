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
    simulation_type = layer_params.get("simulation_type", "default_type")
    fc_layers       = layer_params.get("network_size")
    freq            = layer_params.get("freq", 0)
    non_linearity_type = layer_params.get("non_linearity_type", layer_params.get("non_linearity"))
    if non_linearity_type is None:
        non_linearity_type = layer_params.get("neuron")
    nudging_mode    = layer_params.get("nudging_mode", None)
    bounds          = layer_params.get("bounds", (0.0, 1.0))
    synapse         = layer_params.get("synapse", "default_synapse")
    gamma_values = layer_params.get("gamma_values", (0.0, 0.0))
    cs_bias         = layer_params.get("cs_bias", 0.0)
    non_lin         = layer_params.get("non_lin", False)
    include_bias    = layer_params.get("include_bias", False)
    
    
    transient_params = simulation_parameters.layer_parameters
    
    
    
    layers = []
    
    # Input Layer
    input_layer = InputLayer(
        n_of_nodes=fc_layers[0],
        freq=freq,
        simulation_type=simulation_type,
        which_layer=0,
        include_bias=include_bias,
    )
    layers.append(input_layer)
    
    if not isinstance(gamma_values, (list, tuple)):
        gamma_values = [gamma_values]
    n_dense_layers = max(len(fc_layers) - 1, 0)
    if len(gamma_values) < n_dense_layers:
        pad_value = gamma_values[-1] if gamma_values else 0.0
        gamma_values = list(gamma_values) + [pad_value] * (n_dense_layers - len(gamma_values))

    for idx in range(n_dense_layers):
        dense_layer = DenseLayer(
            n_of_inputs=fc_layers[idx],
            n_of_outputs=fc_layers[idx + 1],
            synapse=synapse,
            bounds=bounds,
            gamma=gamma_values[idx],
            initializer=weight_initializer,
            freq=freq,
            simulation_type=simulation_type,
            which_layer=idx,
        )
        dense_layer.initialize_W()
        layers.append(dense_layer)

        if idx < n_dense_layers - 1:
            non_linear_layer = NonLinearLayer(
                n_of_nodes=fc_layers[idx + 1],
                non_linearity_type=non_linearity_type,
                cs_bias=cs_bias,
                non_lin=non_lin,
                which_layer=idx,
                freq=freq,
                simulation_type=simulation_type,
            )
            layers.append(non_linear_layer)

    if n_dense_layers > 0:
        output_layer = OutputLayer(
            n_of_nodes=fc_layers[-1],
            freq=freq,
            simulation_type=simulation_type,
            nudging_mode=nudging_mode,
            cs_bias=cs_bias,
            which_layer=n_dense_layers - 1,
        )
        layers.append(output_layer)
    
    return layers
