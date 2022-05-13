from abc import ABC, abstractmethod
from scrambled_states import get_scrambled_deepcubea

class Puzzle(ABC):
    @abstractmethod
    def __init__(self, n, state_generator=get_scrambled_deepcubea):  
        pass
          
    @abstractmethod
    def goal_state(self):
        pass
    
    @abstractmethod
    def valid_actions(self, state):
        pass
        
    @abstractmethod
    def number_actions(self, state):
        pass
    
    @abstractmethod
    def get_scrambled(self, B, K):
        pass
        
    @abstractmethod
    def action(self, prev_state, a):
        pass
        
    @abstractmethod
    def inv_action(self, prev_state, a):
        pass
    
    def actions(self, state, acts, return_list = False):
        lst = [state]
        for a in acts:
            state = self.action(state, a)
            if return_list:
                lst.append(state)
        return state if not return_list else lst
        
    @abstractmethod
    def cost(self, state, a):
        pass
    
    @abstractmethod
    def state_str(self, state):
        pass
        
    @abstractmethod
    def get_state_size(self):
        pass
    
    @abstractmethod
    def hash_state(self, state):
       pass
    
    @abstractmethod
    def from_hash(self, bytes_hash): # inverse operation of hash_state
        pass
        
    @abstractmethod
    def __str__(self):
        pass
