from enum import Enum
from itertools import cycle

class SupportedMove(str, Enum):
    FORWARD = "Forward"
    BACKWARD = "Backward"
    TURN_LEFT = "TurnLeft"
    TURN_RIGHT = "TurnRight"
    
def get_turns(start, end):
    supported_moves = []
    
    right = turn_count(cycle([1,2,3,4,5,6,7,8]), start, end)
    left = turn_count(cycle([8,7,6,5,4,3,2,1]), start, end)
    
    if left < right:
        supported_moves.extend([SupportedMove.TURN_LEFT] * left)
    elif right < left:
        supported_moves.extend([SupportedMove.TURN_RIGHT] * right)
    else:
        supported_moves.extend([SupportedMove.TURN_LEFT] * left)
    
    return supported_moves
        
def turn_count(cycle_data, start, end):
    i = 0
    
    for data in cycle_data:
        if i == 0 and data == start:
            i += 1
        elif i > 0 and data == end:
            return i
        elif i > 0:
            i += 1
