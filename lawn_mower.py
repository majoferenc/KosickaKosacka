import dijkstra as dijkstra
from supported_move import SupportedMove
from point import Point
from sensor_response import SensorResponse
import dfs as dfs
import bfs as bfs

def moves_to_exectute(map):
    if map.get_charger_position() is None:
        #use bfs
        bfs.bfs(map.get_current_position(), map)
    else:
        #use dfs
        dfs.dfs(map)
