from queue import PriorityQueue
import numpy as np
import os
import time
from training import Dataset

def clear():
    os.system('clear' if os.name =='posix' else 'cls')
    print("\n"*10)

# f(x) = lambda * g(x) + heuristic(x)
# Expanding N lowest cost nodes in parallel
# OPEN (priority queue)
# CLOSED when node is expanded. Children that are not in CLOSED are added to OPEN
# if we encounter a node x that is already in CLOSED, and if x has a lower path cost
#   than the node that is already in CLOSED, we remove that node from CLOSED and
#   add x to OPEN

# OPEN priority queue
# Costs: dict with state -> (cost, action, parent_state?) (the action from parent_state to state)


def bwas(puzzle, state_0, h, N=5, lamda=1, n_processes=4): 
    open_states = PriorityQueue()
    found = False
    # f(x) = lamda*g(x) + h.predict(x) where g(x) is the cost of the path to x and h(x) is the heuristic 
    g_score = {puzzle.hash_state(state_0): (0, "")} # state -> (cost, action)   (the action from parent_state to state)
    open_states.put((0 + h.predict(np.array([state_0])), 0, state_0)) # PriorityQueue sorts lexicographically
    count = 1 # To break any possible ties.
    if np.array_equal(state_0, puzzle.goal_state()):
        found = True
    num_expanded = 1
    while not found and not open_states.empty():
        states_to_expand = []
        i = 0
        while i < N and not found and not open_states.empty(): # BWAS: expand N states instead of 1
            f_score, _,  state = open_states.get() 
            num_expanded += 1
            if np.array_equal(state, puzzle.goal_state()):
                found = True
                break
            states_to_expand.append(state)
            i += 1
        if found:
            break
        
        child_states = []
        for state in states_to_expand:
            child_states.extend([puzzle.action(state,a) for a in puzzle.valid_actions(state)])
        # Heuristic is computed in parallel for all the childs of the N nodes
        d = Dataset(np.array(child_states), 32) # used for multiprocess predicting
        heuristics = h.predict(np.array(child_states), workers=n_processes, use_multiprocessing=True)
        i = 0
        for state in states_to_expand: 
            actions = puzzle.valid_actions(state)
            num_possible_actions = len(actions)
            new_g_score = g_score[puzzle.hash_state(state)][0] + 1
            for j, a in enumerate(actions): # Each of the N states has len(action) childs
                child = child_states[i + j]
                child_h = heuristics[i + j]
                key = puzzle.hash_state(child)
                if not key in g_score or g_score[key][0] > new_g_score: # PriorityQueue can't update priorities, so this adds repeated states
                    g_score[key] = (new_g_score, a)
                    open_states.put((lamda*new_g_score + child_h, count, child))
                    count += 1
            i += num_possible_actions
        
    state = puzzle.goal_state()
    length, a = g_score[puzzle.hash_state(state)]
    # Build solution
    actions = []
    while a != "":
        actions.append(a)
        state = puzzle.inv_action(state, a)
        _, a = g_score[puzzle.hash_state(state)]
    print(f"     Expanded {num_expanded} states")
    return length, actions[::-1]
    
def animation(puzzle, state, actions):
    for a in actions:
        clear()
        print(puzzle.state_str(state))
        state = puzzle.action(state, a)
        time.sleep(1)
    clear()
    print(puzzle.state_str(state))
    time.sleep(1)
