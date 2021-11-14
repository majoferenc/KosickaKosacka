import time
import requests

TEAM_NAME = "KošickeKosačky@IBM"

def step(session_id, move, base_url):
    if hasattr(move, 'value'):
        # print("Calling step endpoint: {} with paramets id: {} and move {}".format(base_url, session_id, move.value))
        response = requests.get(str(base_url)+"step/", params={"id": session_id, "move": move.value})
    else:
        # print("Calling step endpoint: {} with paramets id: {} and move {}".format(base_url, session_id, move))
        response = requests.get(str(base_url)+"step/", params={"id": session_id, "move": move})        
    # Hack for not crashing api
    time.sleep(0.005)
    # print(response.json())
    return response.json(), response.status_code


def init_session(map_name, base_url):
    # print("Calling init endpoint: {} with paramets mapName: {} and teamName: {}".format(base_url, map_name, TEAM_NAME))
    response = requests.get(str(base_url)+"init/", params={"map": map_name, "team": TEAM_NAME})
    # print(response.json())
    return response.json(), response.status_code
