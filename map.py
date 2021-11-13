from enum import Enum
from point import Point
from sensor_response import SensorResponse

class PositionState(int, Enum):
    OBSTACLE = 1
    BORDER = 2
    # None or Cut
    GRASS = 3
    CHARGER = 4

class Map:
    def __init__(self):
        self.map = {}
        self.charger = None
        self.position = Point(0,0)
        self.direction = [0, 0]

    def update_position(self, direction, position_state: PositionState):
        self.position.X += direction[0]
        self.position.Y += direction[1]

        self.direction = [direction[0], direction[1]]

        self.map[Point(self.position.X, self.position.X)] = position_state

        if (position_state is PositionState.CHARGER):
            self.set_charger_position(Point(self.position.X, self.position.Y))

    def update_position_from_sensor(self, direction, sensor_response: SensorResponse):
        switcher = {
            SensorResponse.NONE:  PositionState.GRASS,
            SensorResponse.OBSTACLE: PositionState.Obstacle,
            SensorResponse.BORDER: PositionState.BORDER,
            SensorResponse.CUT: PositionState.GRASS, 
            SensorResponse.OUT_OF_BOUNDARIES: PositionState.NONE,
            SensorResponse.STUCK: PositionState.NONE,
            SensorResponse.CHARGE: PositionState.CHARGER
        }
        return self.update_position(self, direction, switcher.get(sensor_response, PositionState.NONE))

    """ directly modify the map, should be used only for mocking """
    def set_pair(self, x, y, position_state):
        self.map[Point(x,y)] = position_state
        if (position_state is PositionState.CHARGER):
            self.set_charger_position(Point(x, y))

    def get_map(self):
        return self.map

    def get_position_state(self, coords: Point):
        return self.map[coords]

    def get_charger_position(self):
        return self.charger

    def set_charger_position(self, charger):
        self.charger = charger

    def get_current_direction(self):
        return self.direction

    def get_current_position(self):
        return self.position

    def get_lawn_mower(self):
        return 

    def is_current_position(self, coords):
        return coords == self.position
    