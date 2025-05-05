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
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits

def plot_moons_data(X, Y):
    """
    Plots the moons dataset using only the first two values of each X[i].

    Parameters:
        X (array-like): Input data array with shape (n_samples, n_features).
        Y (array-like): Labels array with shape (n_samples,).
    """
    # Extract the first and second values from each X[i]
    X_first_two = X[:, :2]
    
    # Split the data by class for better visualization
    pos_indices = (Y > 0).flatten()
    neg_indices = (Y <= 0).flatten()
    plt.figure(figsize=(8, 6))
    # Plot positive class
    plt.scatter(X_first_two[:,0][pos_indices], X_first_two[:,1][pos_indices],
                color='blue', label=f'Positive Class ', alpha=0.7)
    # Plot negative class
    plt.scatter(X_first_two[:,0][neg_indices], X_first_two[:,1][neg_indices],
                color='red', label=f'Negative Class ', alpha=0.7)
    
    plt.title('Moons Dataset Visualization (First Two Features)')
    plt.xlabel('X[0]')
    plt.ylabel('X[1]')
    plt.legend()
    plt.grid(True)
    plt.show()


 
def prepare_moons_data(n_samples, noise=0.1, random_state=42):
    # Generate the moons dataset
    X, Y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Center X so that each feature has mean ~ 0
    #X_centered = X - np.mean(X, axis=0)
    

    
    return X, Y

def prepare_iris_data(random_state=42):
    iris_data = load_iris()
    X = iris_data['data']   # shape: (n_samples, n_features)
    Y = iris_data['target'] # shape: (n_samples,)
    
    
    return X, Y

def prepare_linear_regression(n_of_samples):
    x1 = np.ones(n_of_samples)
    x2 = np.ones(n_of_samples)
    y1 = 0.15 * x1 + 0.2 * x2
    y2 = 0.25 * x1 + 0.1 * x2
    X = np.column_stack((x1,x2))
    Y = np.column_stack((y1,y2))
    return X, Y

def prepare_digits_data():
    """
    Loads the digits dataset and returns the features and labels.
    
    Returns:
        X (ndarray): Feature data.
        y (ndarray): Labels.
    """
    digits = load_digits()
    X = digits.data
    y = digits.target
    return X, y

def prepare_simple_dataset(n_samples):
    X = np.ones((n_samples, 3))
    X[:, 0] = -0.2
    X[:, 1] = -0.1
    X[:, 0] = 0
    Y = np.zeros((n_samples, 3))
    Y[:, 2] = 1
    return X, Y


def prepare_wine_data(random_state=42):
    """
    Loads and centers the Wine dataset.
    
    Parameters:
        random_state (int): Random seed for reproducibility (not used in load_wine but kept for interface consistency).
        
    Returns:
        X_centered (numpy.ndarray): The feature matrix with each feature centered (mean ~ 0).
        Y (numpy.ndarray): The target labels.
    """
    # Load the wine dataset
    wine_data = load_wine()
    X = wine_data['data']   # shape: (n_samples, n_features)
    Y = wine_data['target'] # shape: (n_samples,)
    
    
    return X, Y
   
    
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
    Y = Y
    return X_in, Y    


def generate_2_bias_pos_neg_inputs(X, Y, scale_factor, bias, output_scale = 1):
    X_pos =  X * scale_factor 
    X_neg = -X * scale_factor
    X_bias_pos = bias*np.ones((X_pos.shape[0],1))
    X_bias_neg = -bias*np.ones((X_pos.shape[0],1))
    X_in = np.hstack((X_pos, X_neg, X_bias_pos, X_bias_neg))
    Y = Y * output_scale
    return X_in, Y   



def pos_inputs_1bias(X, bias):
    # Scale positive and negative inputs
    X_pos = X
    X_bias_pos = bias * np.ones((X_pos.shape[0], 1))
    
    # Combine inputs
    X_in = np.hstack((X_pos, X_bias_pos))
    
    # Scale output

    # Only one-hot encode if Y is not already one-hot encoded.

    return X_in


def onehot_pos_neg_inputs_1bias(X, Y, scale_factor, bias, output_scale=1):
    # Scale positive and negative inputs
    X_pos = X * scale_factor
    X_neg = -X * scale_factor
    X_bias_pos = bias * np.ones((X_pos.shape[0], 1))
    
    # Combine inputs
    X_in = np.hstack((X_pos, X_neg, X_bias_pos))
    
    # Scale output
    Y_scaled = Y * output_scale

    # Only one-hot encode if Y is not already one-hot encoded.
    # If Y is a 2D array with more than one column, assume it is already one-hot encoded.
    if isinstance(Y_scaled, np.ndarray) and Y_scaled.ndim == 2 and Y_scaled.shape[1] > 1:
        Y_one_hot = Y_scaled
    else:
        # Ensure Y is a 1D array of labels (handles both 1D arrays and (n_samples,1) shaped arrays)
        Y_labels = Y_scaled.flatten()
        num_classes = int(np.max(Y_labels)) + 1  # Assuming classes start from 0
        Y_one_hot = np.eye(num_classes)[Y_labels.astype(int)]
    
    return X_in, Y_one_hot


def onehot_pos_neg_inputs_1bias_double_input(X, Y, bias, output_scale=1):
    # Scale positive and negative inputs
    X_duplicated = np.repeat(X, 1, axis = 0)
    X_pos = X_duplicated
    X_neg = -X_duplicated
    X_bias_pos = bias * np.ones((X_duplicated.shape[0], 1))
    
    # Combine inputs
    X_in = np.hstack((X_pos, X_neg, X_bias_pos))
    
    # Scale output
    Y_duplicated = np.repeat(Y, 1, axis = 0)


    
    # Only one-hot encode if Y is not already one-hot encoded.
    # If Y is a 2D array with more than one column, assume it is already one-hot encoded.
    if isinstance(Y_duplicated, np.ndarray) and Y_duplicated.ndim == 2 and Y_duplicated.shape[1] > 1:
        Y_one_hot = Y_duplicated
    else:
        # Ensure Y is a 1D array of labels (handles both 1D arrays and (n_samples,1) shaped arrays)
        Y_labels = Y_duplicated.flatten()
        num_classes = int(np.max(Y_labels)) + 1  # Assuming classes start from 0
        Y_one_hot = np.eye(num_classes)[Y_labels.astype(int)]
    
    return X_in, Y_one_hot



def onehot_pos_neg_4inputs_doublebias(X, Y, scale_factor, bias, output_scale=1):
    # Scale positive and negative inputs
    X_pos = X * scale_factor
    X_neg = -X * scale_factor
    X_bias_pos = bias * np.ones((X_pos.shape[0], 1))
    X_bias_neg = - bias * np.ones((X_pos.shape[0], 1))
    
    X_pos2 = scale_factor - X_pos
    X_neg2 = -X_pos2
    # Combine inputs
    X_in = np.hstack((X_pos, X_neg, X_pos2, X_neg2, X_bias_pos, X_bias_neg))
    
    # Scale output
    Y_scaled = Y * output_scale

    # Check if Y is already one-hot encoded:
    # If Y is a 2D array with more than one column, assume it is already one-hot encoded.
    if isinstance(Y_scaled, np.ndarray) and Y_scaled.ndim == 2 and Y_scaled.shape[1] > 1:
        Y_one_hot = Y_scaled
    else:
        # Convert Y to one-hot encoding assuming Y contains class labels (e.g., shape (n_samples,) or (n_samples,1))
        Y_labels = Y_scaled.flatten()  # Ensure Y is a 1D array of labels
        num_classes = int(np.max(Y_labels)) + 1  # Assuming classes start from 0
        Y_one_hot = np.eye(num_classes)[Y_labels.astype(int)]
    
    return X_in, Y_one_hot


def linear_regression(n_of_samples,a, b):
    X1 = a * np.ones((n_of_samples, 1))
    X2 = b * np.ones((n_of_samples, 1))
    X = np.hstack((X1, X2))
    Y = np.hstack((0 * np.ones((n_of_samples, 1)), 1 * np.ones((n_of_samples, 1))))
    return X, Y

def onehot_pos_neg_4inputs_singlebias(X, Y, scale_factor, bias, output_scale=1):
    # Scale positive and negative inputs
    X_pos = X 
    X_neg = -X 
    X_bias_pos = bias * np.ones((X_pos.shape[0], 1))
    #X_bias_neg = - bias * np.ones((X_pos.shape[0], 1))
    
    X_pos2 = - X_pos
    X_neg2 = -X_pos2
    # Combine inputs
    X_in = scale_factor * np.hstack((X_pos, X_neg, X_pos2, X_neg2, X_bias_pos))
    
    # Scale output
    Y_scaled = Y * output_scale

    # Check if Y is already one-hot encoded:
    # If Y is a 2D array with more than one column, assume it is already one-hot encoded.
    if isinstance(Y_scaled, np.ndarray) and Y_scaled.ndim == 2 and Y_scaled.shape[1] > 1:
        Y_one_hot = Y_scaled
    else:
        # Convert Y to one-hot encoding assuming Y contains class labels (e.g., shape (n_samples,) or (n_samples,1))
        Y_labels = Y_scaled.flatten()  # Ensure Y is a 1D array of labels
        num_classes = int(np.max(Y_labels)) + 1  # Assuming classes start from 0
        Y_one_hot = np.eye(num_classes)[Y_labels.astype(int)]
    
    return X_in, Y_one_hot



def generate_const_biased_pos_neg_inputs(X, Y, scale_factor, bias):
    X_pos =  X * scale_factor 
    X_neg = -X * scale_factor
    X_pos2 = 1 - X_pos
    X_neg2 = 1 - X_neg
    X_bias_pos = bias*np.ones((X_pos.shape[0],1))
    X_bias_neg = -bias*np.ones((X_pos.shape[0],1))
    X_in = np.hstack((X_pos, X_neg, X_pos2, X_neg2, X_bias_pos, X_bias_neg))
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






