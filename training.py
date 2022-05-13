import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import random, math
import pickle
from queue import Queue
from cube import Cube
import neural_network as nn
from multiprocessing import Process
from optimal import get_optimal_2

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

# ~ def next_cost_old(model_e, X, goal_state, puzzle):
    # ~ y = []
    # ~ for state in X:
        # ~ if np.array_equal(state, goal_state):
            # ~ y.append(0)
            # ~ continue
        # ~ next_states = [puzzle.action(state,a) for a in puzzle.valid_actions(state)]
        # ~ if list(filter(lambda s: np.array_equal(s, goal_state), next_states)):
            # ~ y.append(1) # Mal, hay que tener en cuenta cost
            # ~ continue
        # ~ preds = model_e.predict(np.array(next_states)) + 1 # Mal, hay que tener en cuenta cost
        # ~ y.append(min(preds)[0])
    # ~ return y
    
class Dataset(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, i):
        batch = self.data[i*self.batch_size:(i+1)* self.batch_size]
        return batch

def next_cost(model_e, X, goal_state, puzzle, batch_size=32):
    """
    Calculates the (k+1)-th estimation of the cost function using Bellman update rule.
    model_e: model we are using to evaluate. Represents the k-th estimation
    X: states of which we obtain the updated estimation.
    goal_state: must be obtained as puzzle.goal_state()
    puzzle: problem we are training, subclass of Puzzle
    """
    y = []
    next_states = []
    for state in X:
        next_states.extend([puzzle.action(state,a) for a in puzzle.valid_actions(state)])
    d = Dataset(np.array(next_states), batch_size) # used for multiprocess predicting
    preds = model_e.predict(d, workers=4, use_multiprocessing=True) + puzzle.cost()
    i = 0
    for state in X:
        num_possible_actions = puzzle.number_actions(state)
        state_next_states = next_states[i:i+num_possible_actions]
        state_preds       = preds[i:i+num_possible_actions]
        i += num_possible_actions
        if np.array_equal(state, goal_state):
            y.append(0)
            continue
        if list(filter(lambda s: np.array_equal(s, goal_state), state_next_states)):
            y.append(puzzle.cost()) 
            continue
        y.append(min(state_preds)[0])
    return y

def train(model, puzzle, save_path, initial_conv_point, initial_it, losses, B, B_val, K, M, C, CC, eps, conv_point_offs,lr):
    """
    model: The keras.Model object we want to train
    puzzle: problem we are training, subclass of Puzzle
    initial_it: defaults to 1, change it in case of training from checkpoint
    B: DAVI Batch size
    B_val: validation Batch size 
    K: Maximum number of scrambles
    M: Training iterations
    C: How often to check for convergence
    CC: every CC convergence points, save the weights to the disk
    eps : Error threshold
    """
    if losses is None:
        losses = {"loss":[], "val_loss": []}
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.MeanSquaredError()
    model_e = keras.models.clone_model(model) # model_e with parameters theta_e, used to predict
    model_e.set_weights(model.get_weights()) # clone_model does not copy weights
    model.compile(
        optimizer=optimizer, loss=loss_fn, metrics=[], weighted_metrics=None,
        run_eagerly=False)
    goal_state = puzzle.goal_state()
    print("\n$$ Training " + model.name + f" from iteration {initial_it} to solve " + str(puzzle) + " $$")
    conv_point = initial_conv_point
    for iteration in range(initial_it, initial_it + M): # AVI
        print(f"### Iteration {iteration} convergence point: {conv_point} ###")
        effective_k = min(conv_point+conv_point_offs, K)
        X = puzzle.get_scrambled((B+effective_k - 1)//effective_k, effective_k)
        X_val = puzzle.get_scrambled((B_val + effective_k - 1)//effective_k, effective_k)
        y = next_cost(model_e, X, goal_state, puzzle)
        y_val = next_cost(model_e, X_val, goal_state, puzzle)

        history = model.fit(
            x=np.array(X), y=np.array(y),
            batch_size=None, epochs=1,
            verbose=2, callbacks=None,
            initial_epoch=0, steps_per_epoch=None,
            validation_data=(np.array(X_val), np.array(y_val)),
            )
        loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]
        losses["loss"].append(loss)
        losses["val_loss"].append(val_loss)
        if iteration % C == 0:
            nn.save_losses(model.name, str(puzzle), iteration, save_path, losses)
            if val_loss < eps:
                model_e.set_weights(model.get_weights())
                if (conv_point-1) % CC == 0:
                    nn.save_weights(model, str(puzzle), iteration, save_path)
                conv_point += 1
#Hanoi 4_3 train_deepcubea("DNN_reduced_0_noBatchNorm", 0, puzz=Hanoi(4,3), B=200, B_val=20, K=2**5, C=2, eps=0.05, initial_conv_point=2)
#Hanoi 4_3 train_deepcubea("DNN_final_2x2x2_nr", 0, puzz=Hanoi(4,3), B=200, B_val=20, K=2**5, C=2, eps=0.05, initial_conv_point=2)

#Hanoi_7_3 train_deepcubea("DNN_0", 0, puzz=Hanoi(7,3), B=400, B_val=40, K=2**7, C=2, CC=10, eps=0.05, conv_point_offs=2)

#Hanoi_10_3 train_deepcubea("DNN_0", 0, puzz=Hanoi(10,3), B=500, B_val=50, K=2**10, C=2, CC=10, eps=0.05, conv_point_offs=2) 10000 iterations and 582 convergences
#Hanoi_10_3 train_deepcubea("DNN_reduced_0", 0, puzz=Hanoi(10,3), B=200, B_val=50, K=2**10, C=1, CC=10, eps=0.07, conv_point_offs=2) Total: 10000 iterations and 342 convergences
#Hanoi_10_3 train_deepcubea("DNN_reduced_0", 0, puzz=Hanoi(10,3), B=200, B_val=50,M=100000, K=2**10, C=1, CC=10, eps=0.1, conv_point_offs=2) 12084 iterations = 2.416.800 convergence point: 842
#Hanoi_10_3 train_deepcubea("DNN_0", 0, puzz=Hanoi(10,3), B=200, B_val=40, K=2**10,M=100000, C=2, CC=10, eps=0.1, conv_point_offs=2) 11238 iterations = 2.247.600 convergence point: 506
#Hanoi_10_3 train_deepcubea("DNN_reduced_0", 0, puzz=Hanoi(10,3), B=200, B_val=50, K=2**10, C=1, CC=10, eps=0.5, conv_point_offs=2) 3572 iterations = 714.400 convergence point: 1011 
#Cube3 with mejora K=30, initial_conv_point=1, B=10020, B_val=1020, K=K, M=10000, C=50, eps=0.05
def train_deepcubea(model_name, iteration, save_path = "saved/", puzz = Cube(3), initial_conv_point=1, B=10020, B_val=1020, K=30, M=10000, C=50, CC=1, eps=0.05, conv_point_offs=1): 
    """ EXPLICAR ARGS
    """
    set_seeds(42)
    model = nn.model_factory(model_name, input_shape=puzz.get_state_size())
    model.summary()
    losses = None
    if iteration > 0:
        nn.load_weights(model, str(puzz), iteration, save_path)
        losses = nn.load_losses(model.name, str(puzz), iteration, save_path)
    train(model, puzz, save_path, initial_conv_point, initial_it=iteration+1, losses=losses, B=B, B_val=B_val, K=K, M=M, C=C, CC=CC, eps=eps, conv_point_offs=conv_point_offs,lr=0.0002)
    
# The output of j θ ( s ) is set to 0 if s is the goal state
# Problem: training initial state

