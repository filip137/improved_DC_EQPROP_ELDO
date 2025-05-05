#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:16:28 2025

@author: filip
"""



if __name__ == "__main__":
    
    simulation_type = "FSST" #FSST OR DC
    dataset = "moons"
    
    if dataset == "moons":
        {
          "simulation_details": {
            "initializer": {
              "init_type": "random_uniform",
              "params": {
                "L": 8e-6,
                "U": 1e-7,
                "g_max": 1e-5,
                "g_min": 5e-7
              }
            },
            "layers": {
              "fully_connected": [5, 12, 4],
              "lower_cond_bound": 1e-7,
              "upper_cond_bound": 1e-4
            },
            "learning_rate_factors": {
              "lr_layer1": 1.0,
              "lr_layer2": 1.0
            },
            "gradient_clip": 1,
            "gamma_values": {
                "layer1": 5e-6,
                "layer2": 5e-7
            },
            "beta": 0.5,
            "loss": {
              "type": "MSE"
            },
            "boundary": 0.5
          },
          "dataset_details": {
            "n_of_epochs": 30,
            "scale_factor": 4,
            "noise": 0,
            "bias": 0,
            "num_samples": 2400,
            "batch_size": 15
          },
          "netw