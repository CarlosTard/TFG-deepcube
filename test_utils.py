import matplotlib.pyplot as plt
import numpy as np
import math
import itertools as it
import timeit, time
import os, sys
import pickle
import neural_network as nn
from sklearn.metrics import accuracy_score

from cube import Cube
from hanoi import Hanoi
from optimal import get_optimal_2, get_optimal
from training import Dataset
from scrambled_states import get_scrambled_deepcubea, get_scrambled_deepcube
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.losses import MeanSquaredError as MSE

from bwas import bwas
optimal_2, optimal_dict = None, None

""" State distribution Tests and plots """

def _plot_density_scrambled_states(B, K, c):
    global optimal_2, optimal_dict
    if optimal_2 is None:
        optimal_2, optimal_dict = get_optimal_2()
        
    states  = get_scrambled_deepcubea(B*K, K, c) # returns B*K states
    states2 = get_scrambled_deepcube(B, K, c) # returns B*K states
    opt_values = optimal_2(states)
    opt_values2 = optimal_2(states2)
    
    labels, counts = np.unique(opt_values, return_counts=True)
    labels2, counts2 = np.unique(opt_values2, return_counts=True)
    
    for lab, count, name in [(labels, counts, "DeepCubeA"), (labels2, counts2, "DeepCube")]:
        freq = count / np.sum(count)
        plt.figure()
        plt.bar(lab, freq, align='center')
        plt.xlabel("Value (distance from goal state)")
        plt.ylabel("Probability")
        plt.title(f"{name} K={K}")
        plt.grid(b=True, axis='y')
        plt.draw()
        plt.savefig(f'doc/{name} K={K}.png')
    
    plt.figure()
    plt.ylabel("Difference of frequencies between methods")
    dict1 = dict(zip(labels, counts))
    dict2 = dict(zip(labels2, counts2))
    new_labels = np.union1d(labels, labels2)
    difference = [dict1.get(v, 0) - dict2.get(v,0) for v in new_labels]
    plt.bar(new_labels , difference, align='center')
    plt.xlabel("Value (distance from goal state)")
    plt.title(f"DeepCubeA - DeeepCube K={K}")
    plt.grid(b=True, axis='y')
    plt.draw()
    plt.savefig(f'doc/DeepCubeA - DeeepCube K={K}.png')
    
def plot_density_different_K(B=4000):
    c = Cube(2)
    for K in range(5,25):
        print("Plotting K=", K)
        _plot_density_scrambled_states(B, K, c)
    
def test_new_scrambled_states(B=4000):
    global optimal_2, optimal_dict
    if optimal_2 is None:
        optimal_2, optimal_dict = get_optimal_2()
    c = Cube(2)
    for effective_k in [2,3,7,10,11,14,20]:
        states = get_scrambled_deepcube((B+effective_k - 1)//effective_k, effective_k, c)
        opt_values = optimal_2(states)
        lab, count = np.unique(opt_values, return_counts=True)
        freq = count / np.sum(count)
        plt.figure()
        plt.bar(lab, freq, align='center')
        plt.xlabel("Value (distance from goal state)")
        
        plt.xticks(np.arange(0, 15, 1))
        plt.ylabel("Probability")
        plt.title(f"effective_k={effective_k}")
        plt.grid(b=True, axis='y')
        plt.draw()
        plt.savefig(f'doc/new_scrambled effective_k={effective_k}.png')
                
""" Goal state generation Tests """

def _solved1():
    # ~ 20.797956734000763 seconds
    return [1 if i==j else 0 for (j, k, i) in it.product(range(6), range(9), range(6))]
      
def _solved_one_hot():
    # ~ 8.826677877001202 seconds
    identities = np.tile(np.eye(6,dtype=int),9)
    return identities.reshape((6*9*6))
    
_state_copy = _solved_one_hot()
def _solved_copy(cube_rep=_state_copy): 
    # ~ 0.33895945100084646 seconds
    return cube_rep.copy()

def test_goal_state_generation(n = 1000000):
    print("Testing ", n, " calls")
    print("_solved1 takes: ", timeit.timeit(_solved1, number=n))
    print("_solved_one_hot takes: ", timeit.timeit(_solved_one_hot, number=n))
    print("_solved_copy takes: ", timeit.timeit(_solved_copy, number=n))

