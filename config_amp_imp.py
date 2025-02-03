{
  "simulation_details": {
    "initializer": {
      "init_type": "glorot",
      "params": {
        "L": 0.0001,
        "U": 0.001,
        "g_max": 1e-3,
        "g_min": 2e-5
      }
    },
    "layers": {
      "fully_connected": [5, 8, 2],
      "lower_cond_bound": 2e-6,
      "upper_cond_bound": 1e-3
    },
    "learning_rate_factors": {
      "lr_layer1": 1.0,
      "lr_layer2": 1.0
    },
    "gamma_values": {
      "layer1": 0.0003,
      "layer2": 0.00003
    },
    "beta": 0.001,
    "loss": {
      "type": "MSE",
      "boundary": 0.1
    },
    "dataset": {
      "n_of_epochs": 10,
      "scale_factor": 3,
      "noise": 0,
      "bias": 0,
      "num_samples": 6400,
      "batch_size": 10
    }
  },
  "network_details": {
    "nonlin_parameters": {
      "WIDTH_NMOS_DIFF_A": "650n",
      "VDD": 3.3,
      "V_CASCODE": 2.9,
      "RS_CD": "10k",
      "RS": "10k",
      "RES_DIFF_AMP": "32k",
      "RD": "100k",
      "R_VCVS_BIAS2": "10MEG",
      "R_VCVS_BIAS1": "10MEG",
      "R_SHUNT": "5k",
      "R_D_DIFF_AMP": "65k",
      "R_CCCS_BIAS2": "15MEG",
      "R_CCCS_BIAS1": "10MEG",
      "LOW_NOISE_OPTION": 0,
      "LENGTH_NMOS_DIFF_A": "650n",
      "LENGH_CASC_2": "3.6u",
      "LENGH_CASC_1": "1.6u",
      "IBIAS_DIFF_A": "90u",
      "IBIAS_CASCODE": "30u",
      "CS2_L": "2u",
      "CS1_W": "1.2u",
      "CD1_W": "4u",
      "CD1_L": "650n",
      "CAP": "1n"
    },
    "AC_biases": {
      "source_dc_bias": 2.3
    },
    "frequency": "1Meg",
    "sample_file": "/home/filip/simulations/aex_files",
    "output_dir": "/home/filip/simulations/aex_files"
  }
}