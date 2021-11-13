from enum import Enum
from point import Point
from sensor_response import SensorResponse
from supported_move import SupportedMove

class PositionState(int, Enum):
    OBSTACLE = 1
    BORDER = 2
    # None or Cut
    GRASS = 3
    CHARGER = 4
    MOWER = 5

   
def convert_sensor_response_to_position_state(sensor_response: SensorResponse):
    switcher = {
        SensorResponse.NONE:  PositionState.GRASS,
        SensorResponse.OBSTACLE: PositionState.Obstacle,
        SensorResponse.BORDER: PositionState.BORDER,
        SensorResponse.CUT: PositionState.GRASS, 
        SensorResponse.OUT_OF_BOUNDARIES: PositionState.NONE,
        SensorResponse.STUCK: PositionState.NONE,
        SensorResponse.CHARGE: PositionState.CHARGER
    }
    return switcher.get(sensor_response, PositionState.NONE)

def convert_move_to_direction(direction, move: SupportedMove):
    around = [[1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1], [0,1]]
    index = around.index(direction)
    if move == SupportedMove.TURN_RIGHT:
        index += 1
        if index >= len(around):
            index = 0
    elif move == SupportedMove.TURN_LEFT:
        index -= 1
        if index < 0: 
            index = len(around) - 1 
    return around[index]

class Map:
    def __init__(self):
        self.map = {}
        self.charger = None
        self.position = Point(0,0)
        self.direction = [0, 1]

    def update_position(self, direction, position_state: PositionState):
        self.position = Point(self.position.X + direction[0], self.position.Y + direction[1])

        self.set_direction(direction)

        self.map[self.position] = position_state

        if (position_state is PositionState.CHARGER):
            self.set_charger_position(Point(self.position.X, self.position.Y))

    def update_position_from_sensor(self, direction, sensor_response: SensorResponse):
        return self.update_position(self, direction, convert_sensor_response_to_position_state(sensor_response))

    def update_position_from_move(self, move: SupportedMove, position_state: PositionState):
        # the mower did not move, it just rotated
        if move == SupportedMove.TURN_LEFT or move == SupportedMove.TURN_RIGHT: 
            self.set_direction(convert_move_to_direction(self.direction, move))
        elif move == SupportedMove.FORWARD:
            self.update_position(self.direction, position_state)
        elif move == SupportedMove.BACKWARD:
            # move into the oposite direction
            original_direction = self.direction
            self.update_position([-original_direction[0], -original_direction[1]], position_state)
            # revert the changes on the mower's direction
            self.set_direction(original_direction)

    """ directly modify the map, should be used only for mocking """
    def set_pair(self, x, y, position_state):
        self.map[Point(x,y)] = position_state
        if (position_state is PositionState.CHARGER):
            self.set_charger_position(Point(x, y))

    def set_direction(self, direction):
        self.direction = [direction[0], direction[1]]

    def get_map(self):
        return self.map

    def get_position_state(self, coords: Point):
        if coords in self.map:
            return self.map[coords]
        return None

    def get_charger_position(self):
        return self.charger

    def set_charger_position(self, charger):
        self.charger = charger

    def get_current_direction(self):
        return self.direction

    def get_current_position(self):
        return self.position

    def get_current_position_state(self):
        return self.map.get(self.position)

    def is_current_position(self, coords):
        return coords == self.position 