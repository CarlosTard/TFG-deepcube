import numpy as np

import tensorflow as tf
from puzzle import Puzzle
from scrambled_states import get_scrambled_deepcube
       
def from_one_hot(state):
    return np.argmax(state.reshape(-1,6), axis=1)

class Hanoi(Puzzle):

    def __init__(self, num_disks, num_towers=3, used_disks="all", state_generator=get_scrambled_deepcube):
        self.num_disks = num_disks
        self.num_towers= num_towers
        self.state_generator = state_generator
        if used_disks == "all":
            used_disks = num_disks
        self.used_disks = used_disks
        # A disk on the target tower is represented as [1,0,..,0], and disk on the initial is [0,..0,1]
        self.goal = np.tile(np.eye(1, num_towers,dtype=int),num_disks).reshape(-1) # For each disk, a one-hot encoding the tower where it is
    
    def goal_state(self):
        return self.goal.copy()
    
    def valid_actions(self, state):
        not_empty_towers = set(np.argmax(state.reshape(-1,self.num_towers), axis=1).tolist())
        empty_towers     = set([i for i in range(self.num_towers)]).difference(not_empty_towers)
        top_disks = np.argmax(state.reshape(self.num_disks,self.num_towers).T==1,axis=1) # index i contains top disk of tower i
        # only used_disks are used, to solve H(n',m) with H(n,m), n' < n
        possible_actions = [[(disk, prev_tow, next_tow) for next_tow in range(self.num_towers) if disk < top_disks[next_tow] or next_tow in empty_towers] 
                                                   for prev_tow, disk in enumerate(top_disks) if prev_tow in not_empty_towers and disk < self.used_disks] 
        result = []
        list(map(result.extend, possible_actions)) # concatenate all sublists of possible_actions
        return result 
        
    def number_actions(self, state):
        return len(self.valid_actions(state))        
        
    def cost(self):
        return 1
    
    def get_state_size(self):
        return self.num_disks * self.num_towers
    
    def get_scrambled(self, B_or_l, K): # B_or_l acts as l if state_generator is get_scrambled_deepcube or as B if state_generator is get_scrambled_deepcubea
        return self.state_generator(B_or_l, K, self)
        
    def action(self, prev_state, a): # A is a tuple (disk, prev_tower, next_tower)
        next_s = prev_state.copy()
        next_s.shape = (self.num_disks, self.num_towers)
        disk, prev_tow, next_tow = a

        top_disks = np.argmax(prev_state.reshape(self.num_disks,self.num_towers).T==1,axis=1)
        not_empty_towers = set(np.argmax(prev_state.reshape(-1,self.num_towers), axis=1).tolist())
        empty_towers     = set([i for i in range(self.num_towers)]).difference(not_empty_towers)
        if prev_tow in empty_towers:
            raise ValueError(f"Moving disk {disk} from {prev_tow} to {next_tow}, but {prev_tow} is empty")
        if next_s[disk][prev_tow] != 1:
            raise ValueError(f"Moving disk {disk} from {prev_tow} to {next_tow}, but it was not on tower {prev_tow}")
        if top_disks[prev_tow] != disk:
            raise ValueError(f"Moving disk {disk} from {prev_tow} to {next_tow}, but {disk} not on top of {prev_tow}")
        if next_tow not in empty_towers and top_disks[next_tow] <= disk:
            raise ValueError(f"Moving disk {disk} from {prev_tow} to {next_tow}, but {next_tow} has disk {top_disks[next_tow]} on top")
        if prev_tow == next_tow:
            raise ValueError(f"Moving disk {disk} from {prev_tow} to {next_tow}, but next_tow == prev_tow")
            
        next_s[disk][prev_tow] = 0
        next_s[disk][next_tow] = 1
        next_s.shape = self.num_disks * self.num_towers
        return next_s
    
    def inv_action(self, prev_state, a):
        return self.action(prev_state, (a[0], a[2], a[1]))
        
    def state_str(self, state):
        where_each_disk = np.argmax(state.reshape(-1,self.num_towers), axis=1)
        towers = [[] for i in range(self.num_towers)]
        for disk, tow in enumerate(where_each_disk):
            towers[tow].append(disk)
        return str(towers)[1:-1]
        
    def hash_state(self, state):
        return np.argmax(state.reshape(-1,self.num_towers), axis=1).tobytes()
        
    def from_hash(self, bytes_hash): # inverse operation of hash_state
        return tf.one_hot(np.frombuffer(bytes_hash, 'int64'), self.num_towers,dtype='int64').numpy().reshape(-1)
    
    def __str__(self):
        return f"hanoi_{self.num_disks}_{self.num_towers}"
