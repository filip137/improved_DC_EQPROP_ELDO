#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:46:53 2024

@author: filip
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import time
import subprocess
import shutil
import glob
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split




def prepare_moons_data(n_samples, noise=0.1, random_state=42):
    # Generate the moons dataset
    X, Y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    Y_column = Y.reshape(-1,1)
    # Split the data: 60% for training, 40% for validation and test
    #X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=random_state)
    
    # Split the remaining 40%: 20% for validation, 20% for test
    #X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=random_state)
    return X, Y_column

    
def generate_biased_inputs(X, Y, scale_factor):
    X_scaled = scale_factor * X
    Y_scaled =  Y
    X_bias = scale_factor * (1-X)
    X_in = np.hstack((X_scaled, X_bias))
    return X_in, Y_scaled

def generate_pos_neg_inputs(X, Y, scale_factor , output_scale = 1):
    X_pos =  X * scale_factor 
    X_neg = -X * scale_factor
    X_in = np.hstack((X_pos, X_neg))
    Y = Y * output_scale
    return X_in, Y    

def generate_biased_pos_neg_inputs(X, Y, scale_factor, output_scale = 1):
    X_pos =  X * scale_factor 
    X_neg = -X * scale_factor
    X_bias_pos = scale_factor * (1-X_pos)
    X_bias_neg = scale_factor * (1-X_neg)
    X_in = np.hstack((X_pos, X_neg, X_bias_pos, X_bias_neg))
    Y = Y * output_scale
    return X_in, Y    




def generate_const_biased_pos_neg_inputs(X, Y, scale_factor, bias):
    X_pos =  X * scale_factor 
    X_neg = -X * scale_factor
    X_bias_pos = bias*np.ones((X_pos.shape[0],1))
    X_bias_neg = -bias*np.ones((X_pos.shape[0],1))
    X_in = np.hstack((X_pos, X_neg, X_bias_pos, X_bias_neg))
    return X_in, Y    

def generate_1_bias_pos_neg_inputs(X, Y, scale_factor, bias, output_scale = 1):
    X_pos =  X * scale_factor 
    X_neg = -X * scale_factor
    X_bias = bias*np.ones((X_pos.shape[0],1))
    X_in = np.hstack((X_pos, X_neg, X_bias))
    Y = Y * output_scale
    return X_in, Y   





def generate_dataset(num_samples, mode):
    np.random.seed(2)
    # Randomly generate currents I1 and I2 within a reasonable range
    if mode == "linear_reg":
        V1 = np.random.uniform(0, 5, num_samples)  
        V2 = np.random.uniform(0, 5, num_samples)  
    if mode == "uniform":
    # Initially create 2-dimensional arrays with half the required samples
        V1 = np.ones((num_samples // 2, 1)) * 5  
        V2 = np.ones((num_samples // 2, 1)) * 5 
    
    # Generate random data and reshape immediately to match V1 and V2's 2D shape
        rndm1 = np.random.uniform(1, 5, num_samples // 2).reshape(-1, 1)
        rndm2 = np.random.uniform(1, 5, num_samples // 2).reshape(-1, 1)
    
    # Vertically stack the original and random data
        V1 = np.vstack((V1, rndm1))
        V2 = np.vstack((V2, rndm2))

    # Flatten the arrays to make them 1-dimensional
        V1 = V1.flatten()
        V2 = V2.flatten()
    if mode == "snapshot":
        V1 = np.linspace(1, 5, num_samples)  
        V2 = np.linspace(1, 5, num_samples)        
    if mode == "zeros":
        V1 = np.random.uniform(1, 5, num_samples)  
        V2 = np.random.uniform(1, 5, num_samples)         
        VD1 = np.zeros(num_samples)  
        VD2 = np.zeros(num_samples)    
    # Calculate VD1 and VD2 based on the given formulas
    VD1 = 0.15 * V1  + 0.20 * V2 
    VD2 = 0.25 * V1  + 0.1 * V2
    # VD2=np.ones((num_samples,1))
    # Combine I1 and I2 into a single input feature matrix, and VD1 and VD2 into a targets matrix
    X = np.column_stack((V1, np.zeros([num_samples,1]), V2)) #node1 node4 node7
    Y = np.column_stack((VD1, VD2))

    return X, Y

def generate_dataset_2input_1output(num_samples):
    V1 = np.ones(num_samples)*5
    V2 = np.ones(num_samples)*0
    X = np.column_stack((V1, V2))
    #Y = V1.reshape(-1,1)
    Y = V1.reshape(-1,1)/2
    return X, Y


def generate_xor_data(num_samples):
    """
    Generate a dataset for the XNOR function with specific encoding:
    0 is encoded as -2 and 1 as 2.
    
    Args:
    num_samples (int): Number of (input, output) pairs to generate.
    
    Returns:
    X (numpy.ndarray): The encoded input pairs.
    y (numpy.ndarray): The corresponding XNOR outputs.
    """
    # Randomly generate 0s and 1s for two inputs
    np.random.seed(137)
    X = np.random.randint(0, 2, size=(num_samples, 2))

    # Apply the encoding: 0 -> -2 and 1 -> 2
    X_encoded = np.where(X == 0, -2, 2)
    
    # Compute the XNOR output
    # XOR is true if both bits are the same
    y = np.not_equal(X[:, 0], X[:, 1]).astype(float)
    y = y.reshape(-1,1)
    # Apply encoding to the output as well: 0 -> -2, 1 -> 2
    return X_encoded, y






