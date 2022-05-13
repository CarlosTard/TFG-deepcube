import random

import pickle, bz2

def get_scrambled_deepcubea(B, K, puzzle): # Generates B states
    res = []
    for i in range(B):
        ki = random.randint(1,K)
        s = puzzle.goal_state()
        for j in range(ki):
            s = puzzle.action(s, random.choice(puzzle.valid_actions(s)))
        res.append(s)
    return res
    
def get_scrambled_deepcube(l, K, puzzle): # Generates l*K states
    res = []
    for i in range(l):
        s = puzzle.goal_state()
        for j in range(K):
            s = puzzle.action(s, random.choice(puzzle.valid_actions(s)))
            res.append(s)
    return res
    
def generate_and_save(B,K,n,puzzle,save_path="saved/"):
    for i in range(n):
        X = get_scrambled_deepcubea(B, K, puzzle)
        with bz2.open(save_path + "states/" + str(puzzle) + "/" + str(i), 'wb') as f:
                pickle.dump(X, f)
        
