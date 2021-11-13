import sys
from point import Point
from sensor_response import SensorResponse

def dijkstra(start_point: Point, map: [Point, SensorResponse], foundCharger):
    # distance
    done = {}
    not_done = {}
    # minimal directions to charger
    directions = {}
    done[start_point] = 0

    for p in map.keys():
        not_done[p] = sys.maxsize

    arround = np.array([[1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1], [0,1]])

    for a in arround:
        neighbour = Point(start_point.X + a[0], start_point.Y+ a[1]);
        not_done.pop(neighbour);
        done[neighbour] = 1;
        directions[neighbour] = Point(-a[0],-a[1])

    while not bool(not_done):
        minimum = sys.maxsize
        minPoint = None;
        for p in not_done.keys():
            if not_done.get(p) < minPoint or minPoint is None:
                minimum = not_done.get(p)
                minPoint = p
        done[minPoint] = not_done.get(minPoint)
        not_done.pop(minPoint)

        for a in arround:
            neighbour = Point(start_point.X + a[0], start_point.Y + a[1])
            if neighbour in not_done:
                if map[neighbour] != SensorResponse.NONE or map[neighbour] != SensorResponse.CUT:
                    not_done.pop(neighbour)
                    continue
                turn_distance = abs(directions[minPoint].X + a[0]) + abs(directions[minPoint].Y + a[1])
                if turn_distance == 2 and ((directions[minPoint].X == 0 and a[0] == 0) or (directions[minPoint].Y == 0 and a[1] == 0)):
                    turn_distance = 4
                if not_done[minPoint] > done[minPoint] + turn_distance + 1:
                    not_done[minPoint] = done[minPoint] + turn_distance +1;
                    directions[minPoint] = Point(-a[0],-a[1])

