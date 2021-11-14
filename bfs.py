import queue
import numpy as np
from point import Point
from map import Map, PositionState


# najdenie nabijacky
def bfs(target_point: Point, map: Map) -> tuple:
    bfs_queue = queue.SimpleQueue()
    directions = {}

    passed = []
    bfs_queue.put(target_point)
    passed.append(target_point)

    arround = np.array([[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]])


    while not bfs_queue.empty():
        point = bfs_queue.get(block=False)

        for i in range(7):
            neighbour = Point(point.X + arround[i][0], point.Y + arround[i][1])
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


