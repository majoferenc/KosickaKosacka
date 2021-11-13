from Point import Point
from sensor_response import SensorResponse
from map import PositionState
from map import Map



def dfs():
    map = Map.get_map()
    pos = map.get_current_position()
    for i in [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]:
        step = Point(pos.X + i[0], pos.Y + i[1])
        # Pozicia este nebola preskumana
        if map.get_position_state(step) is None:
            return step

    return None

