import queue
import logging
import numpy as np
import matplotlib.pyplot as plt
from point import Point
from map import Map, PositionState


# najdenie nabijacky
def bfs(start_point: Point, map: Map) -> tuple:
    bfs_queue = queue.Queue()
    directions = {}
    passed = []
    bfs_queue.put(start_point)
    passed.append(start_point)

    arround = np.array([[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]])

    while not bfs_queue.empty():
        logging.debug('Start of while loop')
        point = bfs_queue.get(timeout=False)
        logging.debug(point)
        logging.debug(map.get_position_state(point))
        if map.get_position_state(point) == PositionState.OBSTACLE or map.get_position_state(
                point) == PositionState.BORDER:
            logging.debug('Found Obstacle or Border, skipping')
            continue
        logging.debug('Starting to calculate directions')
        for i in range(7):
            logging.debug(str(i) + '. iteration')
            neighbour = Point(start_point.X + arround[i][0], start_point.Y + arround[i][1])
            position_state = map.get_map().get(neighbour, None)
            logging.debug("Map: " + str(position_state))
            if position_state is PositionState.CHARGER:
                directions[neighbour] = Point(point.X - neighbour.X, point.Y - neighbour.Y)
                logging.debug('Adding to directions 1: ' + str(neighbour) + str(
                    Point(point.X - neighbour.X, point.Y - neighbour.Y)))
                break
            logging.debug('neighbour:' + str(neighbour))
            logging.debug('passed:' + str(passed))
            if neighbour not in passed:
                logging.debug('neighbour not in passed')
                bfs_queue.put(neighbour)
                passed.append(neighbour)
                directions[neighbour] = point
                logging.debug('Adding to directions 2: ' + str(neighbour) + str(point))
        logging.debug('Directions calculated')
        return directions


# Testing BFS
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    start_point: Point = Point(0, 1)
    map_mock: Map = Map()
    map_mock.set_pair(0, 0, PositionState.GRASS)
    map_mock.set_pair(0, 1, PositionState.GRASS)
    map_mock.set_pair(0, 2, PositionState.GRASS)
    map_mock.set_pair(1, 0, PositionState.OBSTACLE)
    map_mock.set_pair(1, 1, PositionState.OBSTACLE)
    map_mock.set_pair(1, 2, PositionState.GRASS)
    map_mock.set_pair(2, 0, PositionState.GRASS)
    map_mock.set_pair(2, 1, PositionState.GRASS)
    map_mock.set_pair(2, 2, PositionState.CHARGER)
    directions = bfs(start_point, map_mock)
    xpoints = []
    ypoints = []
    for direction in directions.items():
        print(type(direction))
        print(direction)
        print(direction[0], direction[1])
        xpoints.append(direction[0].X)
        ypoints.append(direction[0].Y)

    plt.plot(xpoints, ypoints, 'o')
    plt.show()