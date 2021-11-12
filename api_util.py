import requests

BASE_URL = "http://169.51.194.78:31798/"
TEAM_NAME = "KošickeKosačky@IBM"

def step(sessionid, move):
    reponse = requests.get(BASE_URL+"step/", params={"id": sessionid, "move": move})
    print(response.json())
    return reponse.json()


def init_session(mapName):
    response = requests.get(BASE_URL+"init/", params={"mapName": mapName, "teamName": TEAM_NAME})
    print(response.json())
    return response.json()
