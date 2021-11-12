import random
import requests

BASE_URL = "http://192.168.0.101:8888/"

done = False
result = {"sensors": None}
move = None
VALID_MOVES = ['Forward', 'Backward', 'TurnLeft', 'TurnRight']

path = []


def reverse_step():
    move = path.pop()
    if move == 'TurnRight':
        move = 'TurnLeft'
    if move == 'TurnLeft':
        move = 'TurnRight'
    if move == 'Forward':
        move = 'Backward'
    if move == 'Backward':
        move = 'Forward'

    reponse = requests.get(BASE_URL + "step/", params={"id": sessionid, "move": move})

    return reponse.json()



def step(sessionid, move):
    path.append(move)
    reponse = requests.get(BASE_URL + "step/", params={"id": sessionid, "move": move})
    print(reponse)

    return reponse.json()


def initsession():
    response = requests.get(BASE_URL + "init/")
    print(response.json())
    return response.json()


def findValidMove():
    for i in range(1, 8):
        move = "TurnRight"
        result = step(sessionid, move)
        move = "Forward"
        result = step(sessionid, move)
        if result["sensors"] not in ["Obstacle", "Border", "Cut"]:
            return True
        else:
            move = "Backward"
            result = step(sessionid, move)
            return False


def dfs():
    global done
    global result
    global move
    global VALID_MOVES
    global path

    while not done:
        # little logic to not cross border or bump to obstacle
        if result["sensors"] in ["Obstacle", "Border", "Cut"]:
            result = reverse_step()

            validMove = findValidMove()
            while not validMove:
                if not path:
                    pass
                    # makeSomeDecision() # TODO: what to do if i am in closed teritory
                result = reverse_step()
                validMove = findValidMove()

        move = "Forward"
        result = step(sessionid, move)
        done = result["done"]
        print(move, result)


sessionid = '7985651476361'  # initsession()['id']

input("Visualization: " + BASE_URL + "visualize/" +
      sessionid + "\nPress Enter to continue...")

dfs()
