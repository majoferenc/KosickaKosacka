import requests

BASE_URL = "http://169.51.194.78:31798/"
TEAM_NAME = "KošickeKosačky@IBM"

def step(session_id, move):
    print("Calling step endpoint with paramets id: {} and move {}".format(session_id, move.value))
    response = requests.get(BASE_URL+"step/", params={"id": session_id, "move": move.value})
    print(response.json())
    return reponse.json()


def init_session(map_name):
    print("Calling init enpoint with paramets mapName: {} and teamName: {}".format(map_name, TEAM_NAME))
    response = requests.get(BASE_URL+"init/", params={"mapName": map_name, "teamName": TEAM_NAME})
    print(response.json())
    return response.json()
