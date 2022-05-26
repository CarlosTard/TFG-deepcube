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


""" Model Tests and plots """

def plot_loss(model_name, puzzle_name, iteration, save_path="saved/", eps=0.05, C=10): 
    plt.figure()
    losses = nn.load_losses(model_name, puzzle_name, iteration, save_path)
    x = [i + 1 for i in range(0, len(losses["loss"]))]
    plt.plot(x, losses["val_loss"], label="val_loss", color='c', zorder=-1)
    plt.plot(x, losses["loss"], label="train_loss",color='m', zorder = 0)

    plt.axhline(eps, ls="dotted", color='r')
    plt.text(eps, 0,'ε',rotation=0)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Mean Squared Error", fontsize=20)
    plt.title(puzzle_name + " " + model_name, fontsize=20)
    plt.legend(loc="upper right")
    
    points_possible_convergence = np.array(losses["val_loss"][C-1::C])
    iterations_convergence = C*( 1 + (points_possible_convergence < eps).nonzero()[0])
    print(f"Evaluation model has been updated {len(iterations_convergence)} times, at iterations: \n{iterations_convergence}")
    plt.scatter(iterations_convergence, [eps]* len(iterations_convergence),zorder=1, s=40, facecolors='none', edgecolors='k', linewidths=2)
    
    plt.yticks(np.arange(min(losses["loss"]), max(losses["loss"])+1, .02))

    plt.xticks(np.arange(0, len(losses["loss"])+1, C*20))
    plt.draw()
    plt.show(block=False)
        
def divide_batches(arr, batch_size):
    return np.array_split(arr, np.ceil(len(arr)/batch_size))
    
def get_cube2_stats(puzz, save_path='saved/'):
    # Saves and loads labeled data for 2x2x2 cube, and the number of states per value
    file_name = save_path + "optimal/stats_2"
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            stats = pickle.load(f)
    else:
        global optimal_2, optimal_dict
        if optimal_2 is None:
            optimal_2, optimal_dict = get_optimal_2()
        optimal_gen = enumerate(optimal_dict.items())
        states_per_value     = [0]*15
        true_vals = []
        states = []
        for i, (k, value) in enumerate(optimal_dict.items()):
            state = puzz.from_hash(k)
            states.append(state)
            true_vals.append(value)
            states_per_value[value] += 1
            if i % 300000 == 0:
                print("Converting hashes to states. Completed: ", i)
        stats = (states, states_per_value, true_vals)
        with open(file_name, 'wb+') as f:
            pickle.dump(stats, f)
    return stats
    
def plot_errors(true_vals, pred_vals, states_per_value, abs_error_per_value=None, true_positives_per_value=None, block=True):
    mean_abs_err = mean_absolute_error(true_vals, pred_vals)
    mean_squared_err = MSE()(true_vals, pred_vals).numpy()
    print("Mean absolute error:", mean_abs_err)
    print("Mean squared error: ", mean_squared_err)
    
    if abs_error_per_value is not None:
        plt.figure()
        plt.bar([i for i in range(15)], [err/num_states if num_states > 0 else -1 for err, num_states in zip(abs_error_per_value, states_per_value)] , align='center')
        plt.xlabel("Value (distance from goal state)")
        plt.ylabel("Mean absolute error")
        plt.yticks(np.arange(0, 8, .5))
        plt.grid(b=True, axis='y')
        plt.draw()
        plt.show(block=block)
        
    if true_positives_per_value is not None:
        acc = accuracy_score(true_vals, pred_vals)
        print("Accuracy: ", acc)
        plt.figure()
        recalls =  [tp/num_states if num_states > 0 else -1 for tp, num_states in zip(true_positives_per_value, states_per_value)]
        print("Recalls: ", recalls)
        plt.bar([i for i in range(15)], recalls , align='center')
        plt.xlabel("Value (distance from goal state)")
        plt.ylabel("Recall")
        plt.yticks(np.arange(0, 1.1, .1))
        plt.grid(b=True, axis='y')
        plt.draw()
        plt.show(block=block)
        
def test_against_optimal(model, puzz):
    absolute_error = 0
    abs_error_per_value = [0]* 15 # calculate error for each distance from goal state
    states_per_value    = [0]*15
    true_vals, pred_vals = [], []
    states = []
    states, states_per_value, true_vals = get_cube2_stats(puzz)
    batched_states = divide_batches(states, 60)
    print("Processing ", len(batched_states), " batches")
    i = 0
    for j, batch in enumerate(batched_states):
        pred_values = model(batch).numpy().reshape(-1)
        pred_vals.extend(pred_values)
        for pred_val in pred_values:
            absolute_error += abs(true_vals[i] - pred_val)
            abs_error_per_value[true_vals[i]] += abs(true_vals[i] - pred_val)
            i += 1
        if j % 1000 == 0:
            print("batch: ", j, " number elements: ", i)
            
    print("Processed ", i, " states")
    print("States per value: ", states_per_value)
    plot_errors(true_vals, pred_vals, states_per_value, abs_error_per_value)


def test_model_2(model_name, iteration, save_path = "saved/"):
    puzz = Cube(2)
    model = nn.model_factory(model_name, input_shape=6*4*6)
    nn.load_weights(model, str(puzz), iteration, save_path).expect_partial()
    test_against_optimal(model, puzz)
    
def test_admissible_consistency(B, K, puzz, model_name, iteration, optimal=None, save_path = "saved/"):
    model = nn.model_factory(model_name, input_shape=puzz.get_state_size())
    nn.load_weights(model, str(puzz), iteration, save_path).expect_partial()
    states = puzz.get_scrambled(B,K)
    n = len(states)
    d = Dataset(np.array(states), 32) # used for multiprocess predicting
    preds = model.predict(d, workers=4, use_multiprocessing=True)
    predicted_cost = preds.reshape(-1)
    if optimal is not None: # Check for admissibility
        true_cost = np.array(optimal(states))
        admissible_states = predicted_cost <= true_cost
        num_admissible = sum(admissible_states)
        print("It does not overestimate the true cost ", num_admissible/n, "% of the time")
        avg_overestimate = np.average(predicted_cost[~admissible_states] - true_cost[~admissible_states])
        print("Average overestimation of the cost: ", round(avg_overestimate,3))
    
    y = []
    next_states = []
    for state in states:
        next_states.extend([puzz.action(state,a) for a in puzz.valid_actions(state)])
    d = Dataset(np.array(next_states), 16) # used for multiprocess predicting
    preds = model.predict(d, workers=4, use_multiprocessing=True)
    i = 0
    next_state_min_preds = []
    goal_state = puzz.goal_state()
    for state in states: # Check for consistency
        num_possible_actions = puzz.number_actions(state)
        state_next_states = next_states[i:i+num_possible_actions]
        state_preds       = preds[i:i+num_possible_actions]
        i += num_possible_actions
        if np.array_equal(state, goal_state):
            next_state_min_preds.append(0)
            continue
        next_state_min_preds.append(min(puzz.cost() + state_preds)[0]) # min(C(state, state') + H(state'))
    next_state_min_preds = np.array(next_state_min_preds)
    consistent_states = predicted_cost <= next_state_min_preds #H(state) <= min(C(state, state') + H(state'))
    num_consistent = sum(consistent_states)
    print("Consistent states: ", num_consistent/n, "%")
        
def test_admissible_consistency_cube_2(B=2000, K=20, save_path = "saved/"):
    global optimal_2, optimal_dict
    if optimal_2 is None:
        optimal_2, optimal_dict = get_optimal_2()
    models_to_test = [("DNN_0", [2080, 3140, 4050, 7550]), ("DNN_noBatchNorm", [1790, 2540, 3290, 6950]), ("DNN_reduced_0", [2440,3850,5880,9900]),
                         ("DNN_reduced_0_noBatchNorm", [2030, 3660,6140, 11080]), ("DNN_final_2x2x2", [1990, 2350, 2680, 3330]),
                          ("DNN_final_2x2x2_nr", [680, 795, 1315, 1490]), ("DNN_final_2x2x2_nrr", [580, 840, 1250, 1640])]
    c = Cube(2)
    for model_name, iterations in models_to_test:
        for it in iterations:
            print("### Testing consistency for " + model_name + " " + str(it) + "###")
            test_admissible_consistency(B, K, c, model_name, it, optimal=optimal_2, save_path = save_path)

def test_admissible_consistency_hanoi(num_disks, num_towers, B=1000, save_path = "saved/"):
    h = Hanoi(num_disks, num_towers)
    optimal, _ = get_optimal(h)
    K= 2**(num_disks+1)
    if num_disks == 4:
        models_to_test = [("DNN_final_2x2x2_nr", [298, 334, 346, 368, 452]), ("DNN_reduced_0_noBatchNorm", [28,30,34,36])]
    elif num_disks == 7:
        models_to_test = [("DNN_0", [1320, 1354, 1424, 1536])]
    elif num_disks == 10:
        models_to_test = [("DNN_reduced_0", [3088, 3373, 3477, 3494, 3524, 3542,3558, 3572])]
        
    for model_name, iterations in models_to_test:
        for it in iterations:
            print("### Testing consistency for " + model_name + " " + str(it) + "###")
            test_admissible_consistency(B, K, h, model_name, it, optimal=optimal, save_path = save_path)

def test_admissible_consistency_cube_3(B=1000, K=30, save_path = "saved/"):
    c = Cube(3)
    models_to_test = [("DNN_0", [350, 500, 700, 1400])]
        
    for model_name, iterations in models_to_test:
        for it in iterations:
            print("### Testing consistency for " + model_name + " " + str(it) + "###")
            test_admissible_consistency(B, K, c, model_name, it, optimal=None, save_path = save_path)

def unit_tests():
    global optimal_2, optimal_dict     
    c = Cube(2)
    if optimal_2 is None:
        optimal_2, optimal_dict = get_optimal_2()
    g = c.goal_state()
    # Some difficult states:
    state8 = c.from_hash(list(optimal_dict.keys())[-3600000])
    state10 = c.from_hash(list(optimal_dict.keys())[-3000000])
    state11 = c.from_hash(list(optimal_dict.keys())[-1000000])
    state12 = c.from_hash(list(optimal_dict.keys())[-100000])
    state13 = c.from_hash(list(optimal_dict.keys())[-50000])
    state14 = c.from_hash(list(optimal_dict.keys())[-1])
    test_states = np.array([g, c.actions(g, ["R"]), c.actions(g,["R","U"]), c.actions(g,["U","R"]), 
        c.actions(g, ["U", "R", "B"]), c.actions(g, ["U", "R", "B", "R"]), c.actions(g, ["U", "R", "B", "R", "U"]), state8, state10, state11, state12, state13, state14])
    print(optimal_2(test_states[:]))
    
    models_to_test = [("DNN_0", [2080, 7550]), ("DNN_noBatchNorm", [1790,6950]), ("DNN_reduced_0_noBatchNorm", [2030,11080])]
    
    for model_name, iterations in models_to_test:
        for it in iterations:
            mod = nn.model_factory(model_name, input_shape=6*4*6)
            nn.load_weights(mod, str(c), it, "saved/").expect_partial()
            print(model_name, "_", it, ":", mod(test_states))
            
            for state in test_states:
                length, actions = bwas(c, state, mod, N=20, lamda=0.5, n_processes=4)
                print("BWAS: ", length, ", ", actions)
