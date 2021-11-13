from Point import Point
from sensor_response import SensorResponse
from map import PositionState
from map import Map



def dfs():
    pos = Map.getMyPosition()
    for i in [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]:


        # TODO: checks if move is valid
        step = (pos.x + i[0], pos.y + i[1])
        if (step not in ["Obstacle", "Border", "Cut"]): # Nemoze to byt ani nabijacka
            return step
        # TODO: checks if move is valid

    return None

