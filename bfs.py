import queue
import numpy as np
from point import Point
from sensor_response import SensorResponse
from map import PositionState


def bfs(start_point: Point, map: [Point, SensorResponse]):
    bfs_queue = queue.Queue()
    # v mape "predchodca" sa uklada nasledujuci bod v najkratsej ceste ku stanici
    predchodca = []
    directions = {}
    passed = {}
    bfs_queue.put(start_point)
    passed[0] = start_point

    arround = np.array([[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]])

    while True:
        point = bfs_queue.empty()
        if map[point] == SensorResponse.CUT or map[point] == SensorResponse.BORDER:
            continue
        for i in range(8):
            neighbour = Point(start_point.X + arround[i][0], start_point.Y + arround[i][1])
            if map[neighbour] == "lawn_mower":
                directions[i] = (neighbour, point)
                return
            if not passed.get(neighbour):
                bfs_queue.put(neighbour)
                passed[i] = neighbour
                predchodca[i] = neighbour, point
    return directions
    # TODO return directions


# Testing BFS
if __name__ == "__main__":
    start_point: Point = Point(0,1)
    map: [Point, SensorResponse] = [[Point(0,1), SensorResponse.NONE], [Point(0,2), SensorResponse.BORDER]]
    bfs(start_point, map)