import queue
import api_util as api
from sensor_response import SensorResponse
from supported_move import SupportedMove
from map import Map, convert_sensor_response_to_position_state
import webbrowser, pyautogui
import lawn_mower as lawn_mower
from map import PositionState


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
        self.executing_moves_to_charger = False
        # FIFO Queue
        self.execute_queue = queue.Queue()
        self.map_n = Map([0, 1])
        self.map_ne = Map([1, 1])
        self.map_real = self.map_n

        self.map_real.set_pair(0, 0, PositionState.GRASS)
    def execute(self):
        if self.render_mode is True:
            webbrowser.open(self.base_url+"visualize/" + self.session_id)
        if self.charger_distance is None:
            self.last_move = SupportedMove.FORWARD
            step_response, response_code = api.step(self.session_id, self.last_move, self.base_url)
            self.update_step_data(step_response)
        while not self.done:
            if not self.map_real.charger_confirmed:
                approx_charger_point = self.map_real.find_charger(self.charger_direction_offset, self.charger_distance)
            # check if not standing on obstacle or border
            # anti dead mode
            if self.sensor == SensorResponse.OBSTACLE or self.sensor == SensorResponse.BORDER:    
                # execute SupportedMove.FORWARD or SupportedMove.BACKWARD
                self.executing_moves_to_charger = False
                self.put_mirrored_last_move_into_queue()
            elif self.map_real.get_charger_position() is not None and self.executing_moves_to_charger is False:
                moves_to_charger = lawn_mower.moves_to_charger(map)
                
                if len(moves_to_charger) + 6 >= self.power_current:
                    self.executing_moves_to_charger = True
                    self.put_data_array_in_queue(moves_to_charger)

            # check if execute queue is empty
            if self.execute_queue.qsize() == 0:
                self.executing_moves_to_charger = False
                # get moves from algorithm, only if no known moves
                self.put_data_array_in_queue(lawn_mower.moves_to_exectute(self.map_real, approx_charger_point))

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
        
        # update power
        if self.sensor == SensorResponse.CHARGE:
            self.power_current = self.power_max
        else:
            self.power_current -= 1
        
        # update map
        position_state = convert_sensor_response_to_position_state(self.sensor)
        self.map_n.update_position_from_move(self.last_move, position_state)
        self.map_ne.update_position_from_move(self.last_move, position_state)


    def put_data_array_in_queue(self, array):
        self.execute_queue = queue.Queue()
        for data in array:
            self.execute_queue.put(data)


    def put_mirrored_last_move_into_queue(self):
        self.execute_queue = queue.Queue()
        if self.last_move == SupportedMove.FORWARD:
            self.execute_queue.put(SupportedMove.BACKWARD)
        elif self.last_move == SupportedMove.BACKWARD:
            self.execute_queue.put(SupportedMove.BACKWFORWARDARD)
        else:
            raise Exception("Move {} don't have mirrored move.".format(self.last_move))
