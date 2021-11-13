import queue
import api_util as api
from sensor_response import SensorResponse
from supported_move import SupportedMove
import webbrowser, pyautogui
import lawn_mower as lawn_mower

class Model:
    def __init__(self, init_json, base_url, render_mode):
        # init response data
        self.session_id = init_json["id"]
        self.power_max = init_json["maxEnergy"]
        self.power_current = self.power_max
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
        # FIFO Queue
        self.execute_queue = queue.Queue()
        
    def execute(self):
        if self.render_mode is True:
            webbrowser.open(self.base_url+"visualize/" + self.session_id)
        while not self.done:
            # TODO check if lawner has enough energy with dijkstra algo
            need_go_to_charger = False
            
            if need_go_to_charger:
                # TODO go to charger, steps from dijkstra
                self.execute_queue = queue.Queue()
                # self.execute_queue = # get new queue data

            # check if not standing on obstacle or border
            # anti dead mode
            if self.sensor == SensorResponse.OBSTACLE or self.sensor == SensorResponse.BORDER:
                # reset execute_queue 
                self.execute_queue = queue.Queue()
                
                # execute SupportedMove.FORWARD or SupportedMove.BACKWARD
                self.put_mirrored_last_move_into_queue()

            # check if execute queue is empty
            if self.execute_queue.qsize() == 0:
                # get moves from algorithm, only if no known moves
                moves_to_execute = lawn_mower.moves_to_exectute(None)
            
                for move in moves_to_execute:
                    self.execute_queue.put(move)

            # get last_move for anti dead move
            self.last_move = self.execute_queue.get()
            # run step by api
            step_response, response_code = api.step(self.session_id, self.last_move, self.base_url)
            self.update_step_data(step_response)
        
        if self.render_mode is True:
            pass
            # Closing browser tab
            # pyautogui.hotkey('ctrl', 'w')
            
            
    def update_step_data(self, step_json):
        self.done = step_json["done"]
        self.reward = step_json["reward"]
        self.sensor = step_json["sensors"]
        self.charger_distance = step_json["chargerLocation"]["distance"]
        self.charger_direction_offset = step_json["chargerLocation"]["directionOffset"]
        if self.sensor is SensorResponse.CHARGE:
            self.power_current = self.power_max
        else:
            self.power_current -= 1
        # TODO update map


    def put_mirrored_last_move_into_queue(self):
        if self.last_move is SupportedMove.FORWARD:
            self.execute_queue.put(SupportedMove.BACKWARD)
        elif self.last_move is SupportedMove.BACKWARD:
            self.execute_queue.put(SupportedMove.BACKWFORWARDARD)
        else:
            raise Exception("Move {} don't have mirrored move.".format(self.last_move))
