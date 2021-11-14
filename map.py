from enum import Enum
from point import Point
from sensor_response import SensorResponse
from supported_move import SupportedMove
import math


class PositionState(int, Enum):
    OBSTACLE = 1
    BORDER = 2
    # None or Cut
    GRASS = 3
    CHARGER = 4
    MOWER = 5


def convert_sensor_response_to_position_state(sensor_response: SensorResponse):
    switcher = {
        SensorResponse.NONE: PositionState.GRASS,
        SensorResponse.OBSTACLE: PositionState.OBSTACLE,
        SensorResponse.BORDER: PositionState.BORDER,
        SensorResponse.CUT: PositionState.GRASS,
        SensorResponse.OUT_OF_BOUNDARIES: None,
        SensorResponse.STUCK: None,
        SensorResponse.CHARGE: PositionState.CHARGER
    }
    return switcher.get(sensor_response, None)


def convert_move_to_direction(direction, move: SupportedMove):
    around = [[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]]
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
    def __init__(self, start_direction):
        self.map = {}
        self.charger = None
        self.position = Point(0, 0)
        self.direction = start_direction
        self.charger_confirmed = False

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
            # move into the opposite direction
            original_direction = self.direction
            self.update_position([-original_direction[0], -original_direction[1]], position_state)
            # revert the changes on the mower's direction
            self.set_direction(original_direction)

    def update_position_from_move_and_sensor(self, move: SupportedMove, sensor_response: SensorResponse):
        return self.update_position_from_move(move, convert_sensor_response_to_position_state(sensor_response))

    """ directly modify the map, should be used only for mocking """

    def set_pair(self, x, y, position_state):
        self.map[Point(x, y)] = position_state
        if (position_state == PositionState.CHARGER):
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
        self.charger_confirmed = True
        self.charger = charger

    def get_current_direction(self):
        return self.direction

    def get_current_position(self):
        return self.position

    def get_current_position_state(self):
        return self.map.get(self.position)

    def is_current_position(self, coords):
        return coords == self.position

    def find_charger(self, direction_offset, distance):
        # ZMENA ZACINA TU
        if self.direction == [0, 1]:
            my_direction_degrees = 0

        if self.direction == [1, 1]:
            my_direction_degrees = -45

        if self.direction == [1, 0]:
            my_direction_degrees = -90

        if self.direction == [1, -1]:
            my_direction_degrees = -135

        if self.direction == [0, -1]:
            my_direction_degrees = 180

        if self.direction == [-1, -1]:
            my_direction_degrees = 135

        if self.direction == [-1, 0]:
            my_direction_degrees = 90

        if self.direction == [-1, 1]:
            my_direction_degrees = 45

        direction_offset = direction_offset + my_direction_degrees
        if direction_offset > 180:
            direction_offset = direction_offset - 360
        if direction_offset < -180:
            direction_offset = direction_offset + 360

        # ZMENA KONCI TU, ZVYSOK JE ROVNAKY

        x = self.position.X
        y = self.position.Y

        if direction_offset == 0:
            self.charger_confirmed = True
            return Point(x, y + distance)

        if direction_offset == 90:
            self.charger_confirmed = True
            return Point(x - distance, y)

        if abs(direction_offset) == 180:
            self.charger_confirmed = True
            return Point(x, y - distance)

        if direction_offset == -90:
            self.charger_confirmed = True
            return Point(x + distance, y)


        delta = int(distance / math.sqrt(2))
        if direction_offset == 45:
            return Point(x - delta, y + delta)

        if direction_offset == 135:
            return Point(x - delta, y - delta)

        if direction_offset == -45:
            return Point(x + delta, y + delta)

        if direction_offset == -135:
            return Point(x + delta, y - delta)

        my_direction = abs(direction_offset)
        was_big = False
        if my_direction > 90:
            my_direction -= 90
            was_big = True

        cos = math.cos(math.radians(my_direction)) * distance
        sin = math.sin(math.radians(my_direction)) * distance

        if was_big:
            delta_x = cos
            delta_y = -sin
        else:
            delta_x = sin
            delta_y = cos

        if direction_offset > 0:
            delta_x = -delta_x

        return Point(int(x + delta_x), int(y + delta_y))
