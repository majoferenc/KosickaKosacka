from enum import Enum

class PositionState(int, Enum):
    OBSTACLE = 1
    BORDER = 2
    # None or Cut
    GRASS = 3
    CHARGER = 4
    MOWER = 5

class Map:
    def __init__(self):
        self.map = {}
        self.charger = None

    def _getPair(self, x, y):
        return '{},{}'.format(x, y)

    def getCharger(self):
        return self.charger

    def getPositionState(self, x, y):
        return self.map.get(self._getPair(x, y))

    def addPair(self,x, y, state):
        pair = self._getPair(x, y)
        value = self.map.get(self._getPair(x, y))

        if value is None:
            self.map[pair] = state
            if state is PositionState.CHARGER:
                self.charger = pair
        else:
            raise Exception('x: {}, y: {} already set'.format(x, y))
