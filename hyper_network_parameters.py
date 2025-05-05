{
  "simulation_hyper_parameters": {
    "gamma_values": {
      "layer1": 5e-6,
      "layer2": 5e-7
    },
    "beta": 0.5,
    "batch_size": 15,
    "loss_function": "MSE"
  },
  "network_initialization_parameters": {},
  "layers": {
    "fully_connected": [5, 12, 4],
    "lower_cond_bound": 1e-7,
    "upper_cond_bound": 1e-4
  },
  "electrical_network_details": {
    "AC_biases": {
      "source_dc_bias_input": 2.7,
      "source_dc_bias_output": 3.3
    },
    "frequency": "1Meg",
    "simulation_type": "FSST",
    "synapse": "resistor",
    "neuron": "amp_ss",
    "network_files": {
      "sample_file": "/home/filip/simulations/aex_files",
      "output_dir": "/home/filip/simulations/aex_files",
      "seed": 40
    },
    "gradient_clip": 1,
    "boundary": 0.5
  },
  
  "dataset_details": {
    "n_of_epochs": 30,
    "scale_factor": 4,
    "noise": 0,
    "bias": 0,
    "num_samples": 2400
  }
}
