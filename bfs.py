import queue
import logging
import numpy as np
from Point import Point
from map import Map
from map import PositionState

logging.basicConfig(level=logging.DEBUG)


# najdenie nabijacky
def bfs(start_point: Point, map: Map):
    bfs_queue = queue.Queue()
    # v mape "predchodca" sa uklada nasledujuci bod v najkratsej ceste ku stanici
    predchodca = []
    directions = {}
    passed = {}
    bfs_queue.put(start_point)
    passed[0] = start_point

    arround = np.array([[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]])

    while True:
        logging.debug('while')
        point = bfs_queue.get()
        logging.debug(point)
        logging.debug(map.get_position_state(point))
        if map.get_position_state(point) == PositionState.GRASS or map.get_position_state(point) == PositionState.BORDER:
            logging.debug('continue')
            continue
        logging.debug('after if')
        for i in range(7):
            logging.debug('range')
            neighbour = Point(start_point.X + arround[i][0], start_point.Y + arround[i][1])
            if map.is_current_position(neighbour):
                directions[i] = (neighbour, point)
                return
            if not passed.get(neighbour):
                bfs_queue.put(neighbour)
                passed[i] = neighbour
                predchodca[i] = neighbour, point
        logging.debug('after range')
        return directions


# Testing BFS
if __name__ == "__main__":
    start_point: Point = Point(0,1)
    map_mock: Map = Map()
    map_mock.set_pair(0,1, PositionState.GRASS)
    map_mock.set_pair(0,2, PositionState.GRASS)
    map_mock.set_pair(0,3, PositionState.GRASS)
    map_mock.set_pair(1,2, PositionState.GRASS)
    map_mock.set_pair(1,4, PositionState.GRASS)
    map_mock.set_pair(1,3, PositionState.GRASS)
    map_mock.set_pair(2,0, PositionState.GRASS)
    map_mock.set_pair(2,1, PositionState.GRASS)
    map_mock.set_pair(2,3, PositionState.CHARGER)
    directions = bfs(start_point, map_mock)
    print(directions)
