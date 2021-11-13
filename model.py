import queue
import api_util as api
from sensor_response import SensorResponse
from supported_move import SupportedMove
import webbrowser, pyautogui

class Model:
    def __init__(self, init_json, base_url, render_mode):
        # init response data
        self.session_id = init_json["id"]
        self.power_max = init_json["maxEnergy"]
        self.steps_limit = init_json["stepsLimit"]
        self.base_url = base_url
        self.render_mode = render_mode

        # step response data
        self.done = False
        self.reward = None
        self.sensor = None
        self.charger_distance = None
        self.charger_direction_offset = None

        # model variables
        self.last_move = None
        self.execute_queue = queue.Queue()
        
    def execute(self):
        if self.render_mode == "True":
            webbrowser.open(self.base_url+"visualize/" + self.session_id)
        while not self.done:
            # check if not standing on obstacle or border
            if self.sensor is SensorResponse.OBSTACLE or self.sensor is SensorResponse.BORDER:
                # reset execute_queue 
                self.execute_queue = queue.Queue()
                
                # execute SupportedMove.FORWARD or SupportedMove.BACKWARD
                put_mirrored_last_move_into_queue()
            
            # check if execute queue is empty
            if self.execute_queue.qsize() == 0:
                # get moves from algorithm
                moves_to_execute = [SupportedMove.FORWARD]
            
                for move in moves_to_execute:
                    self.execute_queue.put(move)
            step_response, response_code = api.step(self.session_id, self.execute_queue.get(), self.base_url)
            self.update_step_data(step_response)
        if self.render_mode == "True":
            # Closing browser tab
            pass
            # pyautogui.hotkey('ctrl', 'w')
            
            
    def update_step_data(self, step_json):
        self.done = step_json["done"]
        self.reward = step_json["reward"]
        self.sensor = step_json["sensors"]
        self.charger_distance = step_json["chargerLocation"]["distance"]
        self.charger_direction_offset = step_json["chargerLocation"]["directionOffset"]
        
    def put_mirrored_last_move_into_queue(self):
        if self.last_move is SupportedMove.FORWARD:
            self.execute_queue.put(SupportedMove.BACKWARD)
        elif self.last_move is SupportedMove.BACKWARD:
            self.execute_queue.put(SupportedMove.BACKWFORWARDARD)
        else:
            raise Exception("Move {} don't have mirrored move.".format(self.last_move))
