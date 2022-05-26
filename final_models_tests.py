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

""" BWAS Tests """

def grid_time_performance(h, puzzle=Cube(2)):
    # We test the execution time of bwas for 100 random states and 6 difficult states
    test_states = get_scrambled_deepcube(5, 20, puzzle)
    if str(puzzle) == 'cube_2':
        global optimal_2, optimal_dict     
        if optimal_2 is None:
            optimal_2, optimal_dict = get_optimal_2()
        for difficult_state in [-3600000, -3000000, -1000000, -100000, -50000, -1]:
            test_states.append(puzzle.from_hash(list(optimal_dict.keys())[-3600000]))
    test_states = np.array(test_states)
    
    for N in [5, 7, 10]:
        for lamda in [0.1, 0.3, 0.5, 0.7, 1]:
            t0 = time.time()
            for state in test_states:
                bwas(puzzle, state, h, N=N, lamda=lamda)
            elapsed_time = time.time() - t0
            print(f"BWAS with h={h.name} N={N} lamda={lamda} took {elapsed_time} sec.")
                
def test_BWAS_against_optimal(model, puzz, num_tests=10000, K=30, N=5, lamda=0.7):
    global optimal_2, optimal_dict
    if optimal_2 is None:
        optimal_2, optimal_dict = get_optimal_2()

    states_per_value    = [0]*15
    true_positives_per_value = [0]*15
    true_vals, pred_vals = [], []
    states = get_scrambled_deepcube(num_tests//K, K, puzz)

    for i, state in enumerate(states):
        length, _ = bwas(puzz, state, model, N=N, lamda=lamda)
        true_vals.append(optimal_2([state])[0])
        pred_vals.append(length)
        assert(true_vals[-1] <= pred_vals[-1])
        true_positives_per_value[true_vals[-1]] += true_vals[-1] == pred_vals[-1]
        states_per_value[true_vals[-1]] += 1
        if i % (num_tests//10) == 0:
            print("Processed state number ", i)
        
    print("States per value: ", states_per_value)
    plot_errors(true_vals, pred_vals, states_per_value, None, true_positives_per_value, block=False)
    
    
def test_final_models_2(model_name, last_it, save_path="saved/", C=10,  eps=0.05):
    # last_it is the last iteration of training of each model. Only relevant for loading convergence poinst from file
    c = Cube(2)

    losses = nn.load_losses(model_name, str(c), last_it, save_path)
    points_possible_convergence = np.array(losses["val_loss"][C-1::C])
    iterations_convergence = C*( 1 + (points_possible_convergence < eps).nonzero()[0])
    print("######### Testing model: " + model_name + " #########")
    for convergence_it in iterations_convergence[-3:]: # Only the last convergence points
        mod = nn.model_factory(model_name, input_shape=6*4*6)
        nn.load_weights(mod, str(c), convergence_it, "saved/").expect_partial()
        print("## Iteration ", convergence_it)
        test_BWAS_against_optimal(mod, c, num_tests=10000, K=30, N=5, lamda=0.7)
        print()
        
def test_final_models_hanoi(num_disks, num_towers, save_path="saved/"):
    # The last iteration of training of each model. Only relevant for loading convergence poinst from file
    if num_disks == 4:
        models_to_test = [("DNN_final_2x2x2_nr", [298, 334, 346, 368, 452]), ("DNN_reduced_0_noBatchNorm", [28,30,34,36])]
    elif num_disks == 7:
        models_to_test = [("DNN_0", [1320, 1354, 1424, 1536])]
    elif num_disks == 10:
        models_to_test = [("DNN_reduced_0", [3524, 3542,3558, 3572])]
    for model, iterations in models_to_test:
        for it in iterations:
            mod = nn.model_factory(model, input_shape=num_disks * num_towers)
            nn.load_weights(mod, f"hanoi_{num_disks}_{num_towers}", it, "saved/").expect_partial()
            print(f"#### model={model} it={it} ####")
            for N in [5, 7]: # Grid of hyperparameters
                for lamda in [0.3, 0.5, 0.7, 1]:
                    # Transfer learning: we can solve Hanoi(subproblem_disks,num_towers) with the model trained for Hanoi(num_disks, num_towers)
                    for subproblem_disks in range(1,num_disks+1): 
                        solved_disk = [1] + [0]*(num_towers-1)
                        unsolved_disk = [0]*(num_towers-1) + [1] 
                        state = np.array(unsolved_disk*(subproblem_disks)+ solved_disk *(num_disks-subproblem_disks)) 
                        t0 = time.time()
                        h = Hanoi(num_disks, num_towers, subproblem_disks)
                        length, path = bwas(h, state, mod, N=N, lamda=lamda)
                        elapsed_time = time.time() - t0
                        print(f"    N={N} lambda={lamda} subproblem_disks={subproblem_disks} optimal_length={2**subproblem_disks-1}")
                        print(f"    found solution in {elapsed_time}s length={length} path={path}")
            print()

