from enum import Enum
from itertools import cycle

class SupportedMove(str, Enum):
    FORWARD = "Forward"
    BACKWARD = "Backward"
    TURN_LEFT = "TurnLeft"
    TURN_RIGHT = "TurnRight"
    
def get_turns(start, end):
    supported_moves = []
    start_number = get_direction_number(start)
    end_number = get_direction_number(end)
    right = turn_count(cycle([1,2,3,4,5,6,7,8]), start_number, end_number)
    left = turn_count(cycle([8,7,6,5,4,3,2,1]), start_number, end_number)
    
    if left < right:
        supported_moves.extend([SupportedMove.TURN_LEFT] * left)
    elif right < left:
        supported_moves.extend([SupportedMove.TURN_RIGHT] * right)
    else:
        supported_moves.extend([SupportedMove.TURN_LEFT] * left)
    
    return supported_moves

def get_direction_number(direction):
    if direction[0] == -1:
        if direction[1] == 1:
            return 8
        elif direction[1] == 0:
            return 7
        elif direction[1] == -1:
            return 6
    elif direction[0] == 0:
        if direction[1] == 1:
            return 1
        elif direction[1] == -1:
            return 5
    elif direction[0] == 1:
        if direction[1] == 1:
            return 2
        elif direction[1] == 0:
            return 3
        elif direction[1] == -1:
            return 4
    raise Exception("Direction x:{} y:{} not exists.".format(direction[0], direction[1]))
    
def turn_count(cycle_data, start, end):
    i = 0
    
    for data in cycle_data:
        if i == 0 and data == start:
            i += 1
        elif i > 0 and data == end:
            return i
        elif i > 0:
            i += 1
