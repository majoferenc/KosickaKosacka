import queue
import math
import numpy as np
from point import Point
from map import Map, PositionState


# najdenie nabijacky
def bfs(target_point: Point, map: Map) -> tuple:
    directions = {}
    around = np.array([[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]])

    x = map.get_current_position().X - target_point.X
    y = map.get_current_position().Y - target_point.Y
    x = int(math.copysign(1, x))
    y = int(math.copysign(1, y))

    for j in range(2):
        passed = []
        passed.append(target_point)
        bfs_queue = queue.SimpleQueue()
        bfs_queue.put(target_point)
        while not bfs_queue.empty():
            point = bfs_queue.get(block=False)
            for i in range(len(around)):
                neighbour = Point(point.X + around[i][0], point.Y + around[i][1])
                x_n = neighbour.X - target_point.X
                y_n = neighbour.Y - target_point.Y
                x_n = int(math.copysign(1, x_n))
                y_n = int(math.copysign(1, y_n))
                if j == 0:
                    if (x_n == -x) or (y_n == -y):
                        continue
                elif x_n == x and y_n == y:
                    continue
                neighbour_state = map.get_position_state(neighbour)
                if neighbour_state == PositionState.GRASS:
                    directions[neighbour] = Point(point.X - neighbour.X, point.Y - neighbour.Y)
                    return directions, neighbour
                if neighbour not in passed:
                    if neighbour_state == PositionState.OBSTACLE or neighbour_state == PositionState.BORDER:
                        continue
                    bfs_queue.put(neighbour)
                    passed.append(neighbour)
                    directions[neighbour] = Point(point.X - neighbour.X, point.Y - neighbour.Y)


