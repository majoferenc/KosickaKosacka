import random
import requests

BASE_URL = "http://169.51.194.78:31798/"

done = False
result = {"sensors": None}
move = None
VALID_MOVES = ['Forward', 'Backward',  'TurnLeft', 'TurnRight']


def step(sessionid, move):
    reponse = requests.get(BASE_URL+"step/", params={"id": sessionid, "move": move})
    return reponse.json()


def initsession():
    response = requests.get(BASE_URL+"init/")
    print(response.json())
    return response.json()


sessionid = initsession()['id']


input("Visualization: " +BASE_URL+"visualize/" +
      sessionid+"\nPress Enter to continue...")
while not done:
    # little logic to not cross border or bump to obstacle
    validmoves_local=VALID_MOVES
    if result["sensors"] in ["Obstacle", "Border"]:
        if move == "Forward":
            validmoves_local=["Backward"]
        elif move == "Backward":
            validmoves_local=["Forward"]

    move=random.choice(validmoves_local)
    result=step(sessionid, move)
    done=result["done"]
    print(move, result)
