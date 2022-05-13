import numpy as np
from cube import Cube
from collections import deque
import os, sys
import pickle
    
def optimal(puzz, file_name):
    s = puzz.goal_state()
    optimal = {puzz.hash_state(s):0}
    q = deque()
    q.append((s,0))
    i = 0
    while len(q) > 0:
        s, cost = q.popleft()
        for a in puzz.valid_actions(s):
            next_s = puzz.action(s, a)
            hash_s = puzz.hash_state(next_s)
            if hash_s not in optimal:
                optimal[hash_s] = cost + 1
                q.append((next_s, cost+1))
        i += 1
        if i % 100000 == 0:
            print(i, cost, len(optimal))
            
    save_optimal(file_name, optimal)
    return optimal
    
def load_optimal(file_name):
    with open(file_name, 'rb') as f:
        losses = pickle.load(f)
    return losses
    
def save_optimal(file_name, optimal):
    with open(file_name, 'wb+') as f:
        pickle.dump(optimal, f)

def get_optimal(puzz, save_path='saved/'):
    file_name = save_path + "optimal/optimal_" + str(puzz)
    if os.path.isfile(file_name):
        optimal_dict = load_optimal(file_name)
    else:
        optimal_dict = optimal(puzz, file_name)
    def optim(states):
        res = []
        for s in states:
            res.append(optimal_dict[puzz.hash_state(s)])
        return res
    return optim, optimal_dict

def get_optimal_2(save_path='saved/'):
    return get_optimal(Cube(2))
