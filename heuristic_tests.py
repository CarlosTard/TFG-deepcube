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


""" Model Tests and plots """

def plot_loss(model_name, puzzle_name, iteration, save_path="saved/", eps=0.05, C=10): # graficamos una orbita,con lineas horizontales para el cjto atractor
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
    
"""
DNN_0 2080
Mean absolute error: 3.4841625520807327 Mean squared error:  13.584277
DNN_0 7550
Mean absolute error: 2.1626346352224046 Mean squared error:  5.934539
2080 + BWAS 0.99409 Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9977401129943503, 0.9830188679245283, 0.9800687285223367, 0.9927431059506531, 1.0, 1.0, -1]
3140 + BWAS 0.99089 Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9976689976689976, 0.9826756496631376, 0.9656311962987442, 0.9873921698739216, 1.0, 1.0, 1.0]
4050 + BWAS 0.98018 Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9710982658959537, 0.9309188464118041, 0.9558723693143245, 1.0, 1.0, -1]
7550 + BWAS 0.97857 Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9875, 0.9269436997319035, 0.9381443298969072, 0.9973890339425587, 1.0, -1]

DNN_noBatchNorm 1790
Mean absolute error: 2.546816239735354  Mean squared error:  7.8931727 
DNN_noBatchNorm 6950
Mean absolute error: 1.7688242286011782 Mean squared error:  4.262365
650  + BWAS 0.974074 acc 10000 states Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9824766355140186, 0.9356060606060606, 0.9141515761234071, 0.966113914924297, 0.9986072423398329, 1.0, -1]
1080 + BWAS 0.982682 acc 10000 states Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9952662721893492, 0.968299711815562, 0.9407894736842105, 0.9680777238029147, 1.0, 1.0, -1]
1790 + BWAS 0.981281 acc 10000 states Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9692737430167597, 0.9386422976501305, 0.9576570218772054, 1.0, 1.0, -1]
2540 + BWAS 0.97697  acc 10000 states Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.992090395480226, 0.9730483271375465, 0.9174496644295302, 0.9484386347131445, 1.0, 1.0, -1]
3290 + BWAS 0.967867 acc 10000 states Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9976359338061466, 0.9727361246348588, 0.8778371161548731, 0.9274360746371804, 0.996031746031746, 1.0, -1]
6950 + BWAS 0.9643643 acc 10000 states Recalls: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9988372093023256, 0.9836223506743738, 0.8840381991814461, 0.8848569434752268, 0.9960681520314548, 1.0, -1]

DNN_reduced_0 2440
Mean absolute error: 3.055869961975477  Mean squared error: 10.722251
DNN_reduced_0 9900
Mean absolute error: 1.9219732263571299 Mean squared error:  4.88339
2440 + BWAS 0.990690 acc 10000 states Recalls: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9915174363807728, 0.962796664528544, 0.9829842931937173, 1.0, 1.0, 1.0]
3850 + BWAS 0.989789 acc 10000 states Recalls: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9792043399638336, 0.9621656881930855, 0.9853658536585366, 1.0, 1.0, -1]
5880 + BWAS 0.98408 acc 10000 states Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9892996108949417, 0.9405405405405406, 0.9568965517241379, 1.0, 1.0, -1]
9900 + BWAS 0.97037 acc 10000 states Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9885931558935361, 0.9133807369101486, 0.9, 0.9960212201591512, 1.0, -1]

DNN_reduced_0_noBatchNorm 2030
Mean absolute error: 2.573397588408431 Mean squared error:  7.949813 
DNN_reduced_0_noBatchNorm 11080
Mean absolute error: 1.668332215100191 Mean squared error:  3.8863404
2030 + BWAS 0.981781 acc 10000 states Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.998876404494382, 0.9796860572483841, 0.9382879893828799, 0.9534227240649259, 1.0, 1.0, -1]
3660 + BWAS 0.97757 acc 10000 states  Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9777992277992278, 0.9147766323024055, 0.9450072358900145, 0.9985994397759104, 1.0, 1.0]
6140 + BWAS 0.9724724 acc 10000 states Recalls: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9779693486590039, 0.8981741573033708, 0.9251453488372093, 0.9945652173913043, 1.0, -1]
11080 + BWAS 0.9755755 acc 10000 states Recalls: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.998805256869773, 0.9961277831558567, 0.9088453747467927, 0.9327385037748799, 0.9923761118170267, 1.0, -1]

DNN_final_2x2x2 3330
Mean absolute error: 2.4683829689767878 Mean squared error:  7.412141
DNN_final_2x2x2 7460
Mean absolute error: 1.8195473576057648 Mean squared error:  4.585741
[  40   70   90  100  120  150  230  510  980 1380 1990 2350 2680 3330]
1990 + BWAS 0.9855 acc Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9975990396158463, 0.9689849624060151, 0.9451219512195121, 0.9811977715877437, 0.9987357774968394, 1.0, -1]
2350 + BWAS 0.99089 acc Recalls: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.997709049255441, 0.9855595667870036, 0.9713010204081632, 0.9807427785419532, 1.0, 1.0, -1]
2680 + BWAS 0.98958 acc Recalls: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9988137603795967, 0.9779270633397313, 0.9675894665766374, 0.9771265189421015, 1.0, 1.0, 1.0]
3330 + BWAS 0.968768 acc Recalls:[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.997624703087886, 0.9703264094955489, 0.9066937119675457, 0.9072096128170895, 0.9962640099626401, 1.0, -1]

DNN_final_2x2x2_nr 1315
Mean absolute error: 3.623809015028946
Mean squared error:  14.8518915
DNN_final_2x2x2_nr 1490
Mean absolute error: 3.4820554159335666
Mean squared error:  13.719587
[  40   45   55   65   80  115  265  315  470  680  795 1315 1490]
795 + BWAS 0.990490 Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9962216624685138, 0.9908256880733946, 0.9859022556390977, 0.9692832764505119, 0.9821561338289962, 1.0, 1.0, 1.0]
1315 + BWAS 0.987687 Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9978308026030369, 0.9794200187090739, 0.9554794520547946, 0.975, 1.0, 1.0, -1]
1490 + BWAS 0.98178 Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9974937343358395, 0.9867256637168141, 0.9760765550239234, 0.9359673024523161, 0.96639231824417, 1.0, 1.0, -1]

DNN_final_2x2x2_nrr
840  + BWAS 0.972972 acc Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9959677419354839, 0.9813953488372092, 0.9528392685274302, 0.9160460392687881, 0.9463917525773196, 1.0, 1.0, -1]
1250 + BWAS 0.96506 acc Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9948320413436692, 0.9700115340253749, 0.9287804878048781, 0.8882602545968883, 0.9384615384615385, 1.0, 1.0, -1]
1640 + BWAS 0.97017 acc Recalls:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9828767123287672, 0.9429347826086957, 0.9053254437869822, 0.9459843638948117, 1.0, 1.0, 1.0]

"""
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
                #animation(c, state, actions)
                print("BWAS: ", length, ", ", actions)
