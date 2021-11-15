import queue
import api_util as api
import dijkstra as dijkstra
from sensor_response import SensorResponse
from supported_move import SupportedMove
from map import Map, convert_sensor_response_to_position_state
import webbrowser, pyautogui
import lawn_mower as lawn_mower
from point import Point
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
        self.map_real = Map([0, 1])
        self.map_ne = Map([1, 1])
        self.map_real.set_pair(0, 0, PositionState.GRASS)
        self.map_ne.set_pair(0, 0, PositionState.GRASS)       
        self.found_new_tile = True
        self.approx_charger_point = None

    def execute(self):
        if self.render_mode is True:
            webbrowser.open(self.base_url+"visualize/" + self.session_id)
        if self.charger_distance is None:
            self.last_move = SupportedMove.FORWARD
            step_response, response_code = api.step(self.session_id, self.last_move, self.base_url)
            self.update_step_data(step_response)

        while not self.done:
            if not self.map_real.charger_confirmed or (self.map_real.get_position_state(self.approx_charger_point) is not None and self.map_real.get_position_state(self.approx_charger_point) != PositionState.CHARGER):
                if self.map_real.charger_confirmed:
                    print("REAL MAP is wrong 2!!!!")
                    self.map_real = self.map_ne
                    self.execute_queue = queue.Queue()
                    self.map_real.charger_confirmed = False
                self.approx_charger_point = self.map_real.find_charger(self.charger_direction_offset, self.charger_distance)
                while self.map_real.get_position_state(self.approx_charger_point) is not None:
                    self.approx_charger_point = dijkstra.dijkstra_to_unexplored_point(self.approx_charger_point, self.map_real)
            # check if not standing on obstacle or border
            # anti dead mode
            if self.sensor == SensorResponse.OBSTACLE or self.sensor == SensorResponse.BORDER:    
                # execute SupportedMove.FORWARD or SupportedMove.BACKWARD
                self.executing_moves_to_charger = False
                self.put_mirrored_last_move_into_queue()
            elif self.map_real.get_charger_position() is not None and self.executing_moves_to_charger is False:
                moves_to_charger = lawn_mower.moves_to_charger(self.map_real, self.found_new_tile)
                path_multiplier = 3
                if self.map_real is self.map_ne:
                    path_multiplier = 1
                if (len(moves_to_charger) + 6) * path_multiplier >= self.power_current:
                    self.executing_moves_to_charger = True
                    self.put_data_array_in_queue(moves_to_charger)

            # check if execute queue is empty
            if self.execute_queue.qsize() == 0:
                self.executing_moves_to_charger = False
                # get moves from algorithm, only if no known moves
                self.put_data_array_in_queue(lawn_mower.moves_to_exectute(self.map_real, self.approx_charger_point))

            # get last_move for anti dead move
            self.last_move = self.execute_queue.get()
            # run step by api
            step_response, response_code = api.step(self.session_id, self.last_move, self.base_url)
            # print('LAST Move: ' + str(self.last_move))
            self.found_new_tile = self.is_found_new_tile()
            self.update_step_data(step_response)
        
        if self.render_mode is True:
            pass
            # Closing browser tab
            # pyautogui.hotkey('ctrl', 'w')
            
    def is_found_new_tile(self):
        current_position = self.map_real.get_current_position()
        current_direction = self.map_real.get_current_direction()
        if self.last_move != SupportedMove.FORWARD:
            return False
        point = Point(current_position.X + current_direction[0], current_position.Y + current_direction[1])
        return self.map_real.get_position_state(point) is None

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
        if self.map_real is self.map_ne:
            self.map_real.update_position_from_move(self.last_move, position_state)
        else:
            if not self.map_ne.update_position_from_move(self.last_move, position_state):
                print("NORTHEAST MAP is wrong!!!!")
                self.map_ne = self.map_real
                self.execute_queue = queue.Queue()

            if not self.map_real.update_position_from_move(self.last_move, position_state):
                print("REAL MAP is wrong!!!!")
                self.map_real = self.map_ne
                self.execute_queue = queue.Queue()
    def put_data_array_in_queue(self, array):
        self.execute_queue = queue.Queue()
        for data in array:
            self.execute_queue.put(data)


    def put_mirrored_last_move_into_queue(self):
        self.execute_queue = queue.Queue()
        if self.last_move == SupportedMove.FORWARD:
            self.execute_queue.put(SupportedMove.BACKWARD)
        elif self.last_move == SupportedMove.BACKWARD:
            self.execute_queue.put(SupportedMove.FORWARD)
        else:
            raise Exception("Move {} don't have mirrored move.".format(self.last_move))
