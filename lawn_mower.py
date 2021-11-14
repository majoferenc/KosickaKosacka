import dijkstra as dijkstra
from supported_move import SupportedMove,  get_turns
from point import Point
from sensor_response import SensorResponse
import dfs as dfs
import bfs as bfs

def moves_to_exectute(map, approx_charger_point):
    if map.get_charger_position() is None:
        #use bfs
        directions, point = bfs.bfs(approx_charger_point, map)
        if point != map.get_current_position():
            directions.update(dijkstra.dijkstra(point, map))
        current_dir = map.get_current_direction()
        return get_supported_moves(map.get_current_position(), Point(current_dir[0], current_dir[1]), approx_charger_point, directions)
    else:
        #use dfs
        directions, point = dfs.dfs(map)
        return get_supported_moves(map.get_current_position(), map.get_current_direction(), point, directions)

def get_supported_moves(current_position, current_direction, target_position, directions):
    supported_moves = []
    next_point = current_position
    while next_point != target_position:
        direction = directions.get(next_point)
        if type(direction) is list:
            print("Go fuck yourself")
        supported_moves.extend(get_turns(current_direction, direction))
        supported_moves.append(SupportedMove.FORWARD)
        next_point = Point(next_point.X + direction.X, next_point.Y + direction.Y)
    return supported_moves

    # for direction in directions:
    #     supported_moves.extend(get_turns(current_direction_temp, direction))
    #     supported_moves.append(SupportedMove.FORWARD)
    #     current_direction_temp = direction

