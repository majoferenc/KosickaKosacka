import dijkstra as dijkstra
from supported_move import SupportedMove
from point import Point
from sensor_response import SensorResponse
import dfs as dfs

def moves_to_exectute(map: [Point, SensorResponse]):
  # TODO call algorithms, get algo value, set it to moves_to_execute
  # TODO run flood fill dfs
  return [SupportedMove.FORWARD]