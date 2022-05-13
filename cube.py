import numpy as np

from enums import Face
import tensorflow as tf
from puzzle import Puzzle
from scrambled_states import get_scrambled_deepcube


def _solved_one_hot(n):
    identities = np.tile(np.eye(6,dtype=int), n*n)
    return identities.reshape((6*n*n*6))
    
def _solved_int(n):
    identities = np.tile(np.arange(6),n*n)
    return identities.reshape((n*n,6)).transpose()
    
# Warning: makes a copy of state    
def from_one_hot(state):
    return np.argmax(state.reshape(-1,6), axis=1)

class Cube(Puzzle):
    #Used to speed up creation of solved cubes
    SUPPORTED_N = [2,3]
    ONE_HOT     = {n:_solved_one_hot(n) for n in SUPPORTED_N}
    # ~ SOLVED_INT  = {n:_solved_int(n) for n in SUPPORTED_N}
    
    # defines the order of the faces in the state representation
    ORDER       = [Face.L, Face.U, Face.F, Face.D, Face.R, Face.B] 
    """
        O 0    Y 1    B 2    W 3    R 4    G 5
    """
    INV_ORDER   = {e:i for i,e in enumerate(ORDER)}
    STR_ORDER   = [Face.U, Face.L, Face.F, Face.R, Face.B, Face.D] # Only used to give the string representation
     
    # Creates the representation of a solved cube
    def __init__(self, n=3, state_generator=get_scrambled_deepcube):
        if n not in self.SUPPORTED_N:
            raise ValueError(f'n={n} is not a valid cube dimension')
        self.n = n
        self.SHAPE = (6, n, n, 6)
        self.state_generator = state_generator
        
        if self.n == 2:
            self.action_list = ["R", "U","B", "R'", "U'", "B'"]
        else:
            self.action_list = ["F", "R", "U", "L", "B", "D", "F'", "R'", "U'", "L'", "B'", "D'"]
    
    def goal_state(self):
        return Cube.ONE_HOT[self.n].copy()
        
    def get_state_size(self):
        return 6*self.n*self.n*6
    
    def valid_actions(self, state):
        return self.action_list
        
    def number_actions(self, state):
        return len(self.action_list)
        
    def cost(self):
        return 1
    
    def get_scrambled(self, B_or_l, K): # B_or_l acts as l if state_generator is get_scrambled_deepcube or as B if state_generator is get_scrambled_deepcubea
        return self.state_generator(B_or_l, K, self)
        
    def _rotate_views(self, view1, view2, view3, view4):
        first = view1.copy()
        view1[:] = view2
        view2[:] = view3
        view3[:] = view4
        view4[:] = first 
        
    def action(self, prev_state, a):
        next_s = prev_state.copy()
        
        rot_face = Face[a[0]]
        is_inverse = len(a) == 2 and a[1] == "'"
        
        next_s.shape = (6,self.n**2,6)
        if rot_face   ==  Face.R:
            views = [next_s[self.INV_ORDER[Face.F]][self.n-1::self.n,:],              # last col            
                     next_s[self.INV_ORDER[Face.D]][self.n-1::self.n,:],              
                     next_s[self.INV_ORDER[Face.B]][(self.n-1) * self.n::-self.n,:],  # first col reversed
                     next_s[self.INV_ORDER[Face.U]][self.n-1::self.n,:]]
        elif rot_face == Face.L:
            views = [next_s[self.INV_ORDER[Face.D]][0::self.n,:],                     # first col
                     next_s[self.INV_ORDER[Face.F]][0::self.n,:],
                     next_s[self.INV_ORDER[Face.U]][0::self.n,:],
                     next_s[self.INV_ORDER[Face.B]][::-self.n,:]]                     # last col reversed
        elif rot_face == Face.F:
            views = [next_s[self.INV_ORDER[Face.L]][self.n-1::self.n,:],             
                     next_s[self.INV_ORDER[Face.D]][0:self.n:1,:],                    # first row
                     next_s[self.INV_ORDER[Face.R]][(self.n-1) * self.n::-self.n,:],  
                     next_s[self.INV_ORDER[Face.U]][:(self.n-1) * self.n-1:-1,:]]     # last row reversed        
        elif rot_face == Face.B:
            views = [next_s[self.INV_ORDER[Face.U]][self.n-1::-1,:],
                     next_s[self.INV_ORDER[Face.R]][::-self.n,:],
                     next_s[self.INV_ORDER[Face.D]][(self.n-1) * self.n::1,:],
                     next_s[self.INV_ORDER[Face.L]][0::self.n,:]]
        elif rot_face == Face.U:
            views = [next_s[self.INV_ORDER[Face.L]][0:self.n:1,:],
                     next_s[self.INV_ORDER[Face.F]][0:self.n:1,:],
                     next_s[self.INV_ORDER[Face.R]][0:self.n:1,:],
                     next_s[self.INV_ORDER[Face.B]][0:self.n:1,:]]
        elif rot_face == Face.D:
            views = [next_s[self.INV_ORDER[Face.F]][(self.n-1) * self.n::1,:],         # last row
                     next_s[self.INV_ORDER[Face.L]][(self.n-1) * self.n::1,:],
                     next_s[self.INV_ORDER[Face.B]][(self.n-1) * self.n::1,:],
                     next_s[self.INV_ORDER[Face.R]][(self.n-1) * self.n::1,:]]
        if is_inverse:
            views.reverse()
        self._rotate_views(*views)
        next_s.shape = self.SHAPE
        
        if is_inverse:   # anti-clockwise
            next_s[self.INV_ORDER[rot_face]] = np.rot90(next_s[self.INV_ORDER[rot_face]],1)
        else:            # clockwise
            next_s[self.INV_ORDER[rot_face]] = np.rot90(next_s[self.INV_ORDER[rot_face]],3)
        next_s.shape = (6*self.n**2*6)
        return next_s
    
    def inv_action(self, prev_state, a):
        if len(a) == 2:
            a = a[0]
        else:
            a = a + "'"
        return self.action(prev_state, a)
        
    """
    def actions(self, action_list, return_list = False, prev=None): 
        if prev is None:
            prev = self.goal_state()
        if isinstance(action_list, str):
            action_list = action_list.split(' ')
        prev_list = [prev]
        for a in action_list:
            prev = self.action(prev, a)
            if return_list:
                prev_list.append(prev)
        return prev_list if return_list else prev
    """
    def state_str(self, state):
        int_rep = from_one_hot(state)
        int_rep.shape = (6,self.n,self.n)
        # state_colors is the colors of the faces in the state, but with the order given by STR_ORDER
        state_colors = [[[self.ORDER[face].value for face in row ] for row in int_rep[self.INV_ORDER[f]]] for f in self.STR_ORDER]
        string = ''
        spaces = 2 + 2*self.n
        for row in state_colors[0]: # up face
            string += " "*spaces
            for c in row:
                string += c + " "
            string += '\n'
        string += '\n'
        
        for row in range(self.n): 
            for f in range(4):  # left, front, right and back faces
                for col in range(self.n):  
                    string += ' '  + state_colors[f + 1][row][col]
                string += ' '
            string += '\n'
        
        string += '\n'    
        for row in state_colors[5]: # down face 
            string += " "*spaces
            for c in row:
                string += c + " "
            string += '\n'
        return string
    
    def hash_state(self, state):
        return np.argmax(state.reshape(-1,6), axis=1).tobytes()
        
    def from_hash(self, bytes_hash): # inverse operation of hash_state
        return tf.one_hot(np.frombuffer(bytes_hash, 'int64'), 6,dtype='int64').numpy().reshape(-1)
    
    def __str__(self):
        return f"cube_{self.n}"
