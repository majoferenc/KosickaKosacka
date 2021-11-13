from enum import Enum
from Point import Point

class PositionState(int, Enum):
    OBSTACLE = 1
    BORDER = 2
    GRASS = 3
    CHARGER = 4

class Map:
    def __init__(self):
        self.map = {}
        self.charger = None
        self.position = Point(0,0)
        self.direction = [0, 0]

    def update_position(self, direction, position_state):
        self.position.X += direction[0]
        self.position.Y += direction[1]

        self.direction = [direction[0], direction[1]]

        self.map[Point(self.position.X, self.position.X)] = position_state

        if (position_state is PositionState.CHARGER):
            self.set_charger_position(Point(self.position.X, self.position.Y))

    def get_map(self):
        return self.map

    def get_position_state(self, coords):
        return self.map[coords]


    def get_charger_position(self):
        return self.charger

    def set_charger_position(self, charger):
        self.charger = charger

    def get_current_direction(self):
        return self.direction

    def get_current_position(self):
        return self.position