import dijkstra as dijkstra
from supported_move import SupportedMove,  get_turns
from point import Point
from sensor_response import SensorResponse
import dfs as dfs
import bfs as bfs

def moves_to_exectute(map):
    if map.get_charger_position() is None:
        #use bfs
        ponts, directions = bfs.bfs(map.get_current_position(), map)
        return get_supported_moves(map.get_current_direction(), directions)
    else:
        #use dfs
        ponts, directions = dfs.dfs(map)
        return get_supported_moves(map.get_current_direction(), directions)

def get_supported_moves(current_direction, directions):
    supported_moves =[]
    current_direction_temp = current_direction
    
    for direction in directions:
        supported_moves.extend(get_turns(current_direction_temp, direction))
        supported_moves.append(SupportedMove.FORWARD)
        current_direction_temp = direction
    
