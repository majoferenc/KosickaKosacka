from enum import Enum

class SupportedMove(str, Enum):
    FORWARD = "Forward"
    BACKWARD = "Backward"
    TURN_LEFT = "TurnLeft"
    TURN_RIGHT = "TurnRight"
