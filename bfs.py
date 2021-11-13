import queue
from point import Point
from sensor_response import SensorResponse
from map import PositionState

def bfs(start_point: Point, map: [Point, SensorResponse]):
  queue = queue.Queue()
  # v mape "predchodca" sa uklada nasledujuci bod v najkratsej ceste ku stanici
  predchodca = []
  directions = {}
  passed = {}
  queue.put(startPoint)
  passed[0] = startPoint

  arround = np.array([[1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1], [0,1]])
  while True:
    point = queue.remove()
    if map[point] == SensorResponse.CUT or map[point] == SensorResponse.BORDER:
      continue
    for i in range(8):
      neighbour = Point(start_point[x] + arround[i][0], start_point[y] + arround[i][1])
      if map[neighbour] == "lawn_mower":
        predchodca.add(neighbour, point)
        return
      if !passed.contais(neighbour):
        queue.put(neighbour)
        passed.add(neighbour)
        predchodca.add(neighbour, point)