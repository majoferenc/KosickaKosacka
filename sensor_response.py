from enum import Enum

class SensorResponse(str, Enum):
    NONE = "None"
    OBSTACLE = "Obstacle"
    BORDER = "Border"
    CUT = "Cut"
    OUT_OF_BOUNDARIES = "OutOfBoundaries"
    STUCK = "Stuck"
    CHARGE = "Charge"
