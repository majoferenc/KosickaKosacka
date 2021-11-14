import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from point import Point
from sensor_response import SensorResponse
from map import Map, PositionState


def dijkstra(destination_point: Point, map: Map):
    done = {}
    not_done = {}
    # minimal directions to charger
    directions = {}
    done[destination_point] = 0
    # directions[destination_point] = Point(0, 0)

    for p in map.get_map():
        not_done[p] = sys.maxsize
    not_done.pop(destination_point, None)
    arround = np.array([[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]])

    for a in arround:
        neighbour = Point(destination_point.X + a[0], destination_point.Y + a[1])
        if neighbour in not_done and \
                (map.get_map().get(neighbour, None) == PositionState.GRASS or \
                map.get_map().get(neighbour, None) == PositionState.CHARGER):
            not_done[neighbour] = 1
            directions[neighbour] = Point(-a[0], -a[1])

    while bool(not_done):

        # for nd in not_done.items():
        #     logging.debug(nd[0], nd[1])

        minimum = sys.maxsize
        minPoint = None
        for p in not_done.keys():
            if minPoint is None or not_done.get(p) < minimum:
                minimum = not_done.get(p)
                minPoint = p
        done[minPoint] = not_done.get(minPoint)
        # logging.debug("Min point" + str(minPoint))
        # logging.debug("---------------------")
        # for nd in done.items():
        #     logging.debug(nd[0], nd[1])
        # logging.debug("===")
        not_done.pop(minPoint, None)

        for a in arround:
            neighbour = Point(minPoint.X + a[0], minPoint.Y + a[1])
            if neighbour in not_done:
                if map.get_map().get(neighbour, None) != PositionState.GRASS and map.get_map().get(neighbour, None) != PositionState.CHARGER:
                    not_done.pop(neighbour, None)
                    continue
                turn_distance = abs(directions[minPoint].X + a[0]) + abs(directions[minPoint].Y + a[1])
                if turn_distance == 2 and (
                        (directions[minPoint].X == 0 and a[0] == 0) or (directions[minPoint].Y == 0 and a[1] == 0)):
                    turn_distance = 4
                if not_done.get(neighbour) > done.get(minPoint) + turn_distance + 1:
                    not_done[neighbour] = done[minPoint] + turn_distance + 1
                    directions[neighbour] = Point(-a[0], -a[1])

    return directions


def dijkstra_to_unexplored_point(destination_point: Point, map: Map):
    done = {}
    not_done = {}
    # minimal directions to charger
    directions = {}
    done[destination_point] = 0
    # directions[destination_point] = Point(0, 0)

    for p in map.get_map():
        not_done[p] = sys.maxsize
    not_done.pop(destination_point, None)
    arround = np.array([[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]])
    for a in arround:
        neighbour = Point(destination_point.X + a[0], destination_point.Y + a[1])
        if neighbour in not_done and \
                (map.get_map().get(neighbour, None) == PositionState.GRASS or \
                 map.get_map().get(neighbour, None) == PositionState.CHARGER):
            not_done[neighbour] = 1
            directions[neighbour] = Point(-a[0], -a[1])
    while bool(not_done):
        # for nd in not_done.items():
        #     logging.debug(nd[0], nd[1])
        minimum = sys.maxsize
        minPoint = None
        for p in not_done.keys():
            if minPoint is None or not_done.get(p) < minimum:
                minimum = not_done.get(p)
                minPoint = p
        done[minPoint] = not_done.get(minPoint)
        # logging.debug("Min point" + str(minPoint))
        # logging.debug("---------------------")
        # for nd in done.items():
        #     logging.debug(nd[0], nd[1])
        # logging.debug("===")
        not_done.pop(minPoint, None)
        for a in arround:
            neighbour = Point(minPoint.X + a[0], minPoint.Y + a[1])

            if map.get_map().get(neighbour, None) == None:
                return directions, neighbour
            if neighbour in not_done:
                if map.get_map().get(neighbour, None) != PositionState.GRASS and map.get_map().get(neighbour,
                                                                                                   None) != PositionState.CHARGER:
                    not_done.pop(neighbour, None)
                    continue
                turn_distance = abs(directions[minPoint].X + a[0]) + abs(directions[minPoint].Y + a[1])
                if turn_distance == 2 and (
                        (directions[minPoint].X == 0 and a[0] == 0) or (directions[minPoint].Y == 0 and a[1] == 0)):
                    turn_distance = 4
                if not_done.get(neighbour) > done.get(minPoint) + turn_distance + 1:
                    not_done[neighbour] = done[minPoint] + turn_distance + 1
                    directions[neighbour] = Point(-a[0], -a[1])


# Testing BFS
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    destination_point: Point = Point(0, 1)
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
    directions = dijkstra(destination_point, map_mock)
    xpoints = []
    ypoints = []
    for direction in directions.items():
        logging.debug(direction[0], direction[1])
        xpoints.append(direction[0].X)
        ypoints.append(direction[0].Y)

    plt.plot(xpoints, ypoints, 'o')
    plt.show()