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

from bwas import bwas, animation
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

"""
>>> test_admissible_consistency_cube_2(2000,20)
### Testing consistency for DNN_0 2080###
Consistent states:  0.708425 %
### Testing consistency for DNN_0 3140###
It does not overestimate the true cost  0.874075 % of the time
Average overestimation of the cost:  0.174
Consistent states:  0.669575 %
### Testing consistency for DNN_0 4050###
It does not overestimate the true cost  0.815725 % of the time
Average overestimation of the cost:  0.235
Consistent states:  0.69255 %
### Testing consistency for DNN_0 7550###
It does not overestimate the true cost  0.790225 % of the time
Average overestimation of the cost:  0.296
Consistent states:  0.6763 %
### Testing consistency for DNN_noBatchNorm 1790###
It does not overestimate the true cost  0.71375 % of the time
Average overestimation of the cost:  0.222
Consistent states:  0.598425 %
### Testing consistency for DNN_noBatchNorm 2540###
It does not overestimate the true cost  0.718275 % of the time
Average overestimation of the cost:  0.298
Consistent states:  0.636325 %
### Testing consistency for DNN_noBatchNorm 3290###
It does not overestimate the true cost  0.5358 % of the time
Average overestimation of the cost:  0.216
Consistent states:  0.5736 %
### Testing consistency for DNN_noBatchNorm 6950###
It does not overestimate the true cost  0.587 % of the time
Average overestimation of the cost:  0.26
Consistent states:  0.545 %
### Testing consistency for DNN_reduced_0 2440###
It does not overestimate the true cost  0.754175 % of the time
Average overestimation of the cost:  0.169
Consistent states:  0.6956 %
### Testing consistency for DNN_reduced_0 3850###
It does not overestimate the true cost  0.6517 % of the time
Average overestimation of the cost:  0.208
Consistent states:  0.63805 %
### Testing consistency for DNN_reduced_0 5880###
It does not overestimate the true cost  0.6948 % of the time
Average overestimation of the cost:  0.246
Consistent states:  0.60025 %
### Testing consistency for DNN_reduced_0 9900###
It does not overestimate the true cost  0.76535 % of the time
Average overestimation of the cost:  0.289
Consistent states:  0.49225 %
### Testing consistency for DNN_reduced_0_noBatchNorm 2030###
It does not overestimate the true cost  0.707925 % of the time
Average overestimation of the cost:  0.304
Consistent states:  0.616075 %
### Testing consistency for DNN_reduced_0_noBatchNorm 3660###
It does not overestimate the true cost  0.70765 % of the time
Average overestimation of the cost:  0.342
Consistent states:  0.613675 %
### Testing consistency for DNN_reduced_0_noBatchNorm 6140###
It does not overestimate the true cost  0.62785 % of the time
Average overestimation of the cost:  0.266
Consistent states:  0.57685 %
### Testing consistency for DNN_reduced_0_noBatchNorm 11080###
It does not overestimate the true cost  0.49705 % of the time
Average overestimation of the cost:  0.195
Consistent states:  0.5979 %
### Testing consistency for DNN_final_2x2x2 1990###
It does not overestimate the true cost  0.8647 % of the time
Average overestimation of the cost:  0.251
Consistent states:  0.5763 %
### Testing consistency for DNN_final_2x2x2 2350###
It does not overestimate the true cost  0.791775 % of the time
Average overestimation of the cost:  0.213
Consistent states:  0.67615 %
### Testing consistency for DNN_final_2x2x2 2680###
It does not overestimate the true cost  0.808125 % of the time
Average overestimation of the cost:  0.29
Consistent states:  0.610075 %
### Testing consistency for DNN_final_2x2x2 3330###
It does not overestimate the true cost  0.7687 % of the time
Average overestimation of the cost:  0.255
Consistent states:  0.667125 %
### Testing consistency for DNN_final_2x2x2_nr 680###
It does not overestimate the true cost  0.744125 % of the time
Average overestimation of the cost:  0.213
Consistent states:  0.65645 %
### Testing consistency for DNN_final_2x2x2_nr 795###
It does not overestimate the true cost  0.6646 % of the time
Average overestimation of the cost:  0.208
Consistent states:  0.701875 %
### Testing consistency for DNN_final_2x2x2_nr 1315###
It does not overestimate the true cost  0.71315 % of the time
Average overestimation of the cost:  0.244
Consistent states:  0.587725 %
### Testing consistency for DNN_final_2x2x2_nr 1490###
It does not overestimate the true cost  0.731725 % of the time
Average overestimation of the cost:  0.243
Consistent states:  0.6621 %
### Testing consistency for DNN_final_2x2x2_nrr 580###
It does not overestimate the true cost  0.831475 % of the time
Average overestimation of the cost:  0.257
Consistent states:  0.6982 %
### Testing consistency for DNN_final_2x2x2_nrr 840###
It does not overestimate the true cost  0.80565 % of the time
Average overestimation of the cost:  0.272
Consistent states:  0.6825 %
### Testing consistency for DNN_final_2x2x2_nrr 1250###
It does not overestimate the true cost  0.7119 % of the time
Average overestimation of the cost:  0.221
Consistent states:  0.6511 %
### Testing consistency for DNN_final_2x2x2_nrr 1640###
It does not overestimate the true cost  0.64005 % of the time
Average overestimation of the cost:  0.225
Consistent states:  0.572925 %
>>> 

"""

"""
>>> test_admissible_consistency_hanoi(4, 3)
### Testing consistency for DNN_final_2x2x2_nr 298###
It does not overestimate the true cost  0.7844375 % of the time
Average overestimation of the cost:  0.294
Consistent states:  0.46678125 %
### Testing consistency for DNN_final_2x2x2_nr 334###
It does not overestimate the true cost  0.5330625 % of the time
Average overestimation of the cost:  0.296
Consistent states:  0.4424375 %
### Testing consistency for DNN_final_2x2x2_nr 346###
It does not overestimate the true cost  0.69490625 % of the time
Average overestimation of the cost:  0.443
Consistent states:  0.51475 %
### Testing consistency for DNN_final_2x2x2_nr 368###
It does not overestimate the true cost  0.49340625 % of the time
Average overestimation of the cost:  0.302
Consistent states:  0.2325625 %
### Testing consistency for DNN_final_2x2x2_nr 452###
It does not overestimate the true cost  0.723 % of the time
Average overestimation of the cost:  0.242
Consistent states:  0.50103125 %
### Testing consistency for DNN_reduced_0_noBatchNorm 28###
It does not overestimate the true cost  0.70075 % of the time
Average overestimation of the cost:  0.103
Consistent states:  0.72634375 %
### Testing consistency for DNN_reduced_0_noBatchNorm 30###
It does not overestimate the true cost  0.921875 % of the time
Average overestimation of the cost:  0.073
Consistent states:  0.85603125 %
### Testing consistency for DNN_reduced_0_noBatchNorm 34###
It does not overestimate the true cost  0.47978125 % of the time
Average overestimation of the cost:  0.053
Consistent states:  0.80734375 %
### Testing consistency for DNN_reduced_0_noBatchNorm 36###
It does not overestimate the true cost  0.6295 % of the time
Average overestimation of the cost:  0.067
Consistent states:  0.416 %
"""

"""
>>> test_admissible_consistency_hanoi(7, 3)
### Testing consistency for DNN_0 1320###
It does not overestimate the true cost  0.93051953125 % of the time
Average overestimation of the cost:  0.084
Consistent states:  0.64762109375 %
### Testing consistency for DNN_0 1354###
It does not overestimate the true cost  0.941703125 % of the time
Average overestimation of the cost:  0.014
Consistent states:  0.59973046875 %
### Testing consistency for DNN_0 1424###
It does not overestimate the true cost  0.906109375 % of the time
Average overestimation of the cost:  0.101
Consistent states:  0.6808359375 %
### Testing consistency for DNN_0 1536###
It does not overestimate the true cost  0.74820703125 % of the time
Average overestimation of the cost:  0.171
Consistent states:  0.61960546875 %

"""
"""
>>> test_admissible_consistency_hanoi(10, 3, B=200)
### Testing consistency for DNN_reduced_0 3088###
It does not overestimate the true cost  0.89654296875 % of the time
Average overestimation of the cost:  0.808
Consistent states:  0.78473388671875 %
### Testing consistency for DNN_reduced_0 3373###
It does not overestimate the true cost  0.80041748046875 % of the time
Average overestimation of the cost:  0.411
Consistent states:  0.75222900390625 %
### Testing consistency for DNN_reduced_0 3477###
It does not overestimate the true cost  0.91769775390625 % of the time
Average overestimation of the cost:  0.817
Consistent states:  0.7343212890625 %
### Testing consistency for DNN_reduced_0 3494###
It does not overestimate the true cost  0.76075927734375 % of the time
Average overestimation of the cost:  0.691
Consistent states:  0.77665771484375 %
### Testing consistency for DNN_reduced_0 3524###
It does not overestimate the true cost  0.81938232421875 % of the time
Average overestimation of the cost:  0.496
Consistent states:  0.705361328125 %
### Testing consistency for DNN_reduced_0 3542###
It does not overestimate the true cost  0.97367431640625 % of the time
Average overestimation of the cost:  0.02
Consistent states:  0.75810546875 %
### Testing consistency for DNN_reduced_0 3558###
It does not overestimate the true cost  0.86241943359375 % of the time
Average overestimation of the cost:  0.321
Consistent states:  0.769208984375 %
### Testing consistency for DNN_reduced_0 3572###
It does not overestimate the true cost  0.9217529296875 % of the time
Average overestimation of the cost:  0.428
Consistent states:  0.81922607421875 %
"""

"""
>>> test_admissible_consistency_cube_3()
### Testing consistency for DNN_0 350###
onsistent states:  0.9395666666666667 %
### Testing consistency for DNN_0 500###
Consistent states:  0.9364333333333333 %
### Testing consistency for DNN_0 700###
Consistent states:  0.9083666666666667 %
### Testing consistency for DNN_0 1400###
Consistent states:  0.8677 %
"""
