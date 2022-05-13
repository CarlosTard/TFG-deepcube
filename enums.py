from enum import Enum
""" 3x3
        Y Y Y
        Y Y Y
        Y Y Y
        
 O O O  B B B  R R R  G G G
 O O O  B B B  R R R  G G G
 O O O  B B B  R R R  G G G
 
        W W W
        W W W
        W W W 
"""

""" 2x2
      Y Y
      Y Y
        
 O O  B B  R R  G G
 O O  B B  R R  G G
 
      W W
      W W
"""

class Color(Enum):
    R = '\033[31m'
    G = '\033[92m'
    Y = '\033[93m'
    B = '\033[94m'
    W = '\033[97m'
    O = '\033[33m'
    END = '\x1b[0m'
    
# Associates each face with a color UP DOWN FRONT RIGHT LEFT BACK
class Face(Enum): 
    U = Color.Y.value + 'Y' + Color.END.value
    D = Color.W.value + 'W' + Color.END.value
    F = Color.B.value + 'B' + Color.END.value
    R = Color.R.value + 'R' + Color.END.value
    L = Color.O.value + 'O' + Color.END.value
    B = Color.G.value + 'G' + Color.END.value

