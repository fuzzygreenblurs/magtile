from constants import *
import math
import numpy as np
import networkx as nx
from agent_color import AgentColor
import pdb

class Agent:
    _actuator = None

    @classmethod
    def set_actuator(cls, actuator):
        cls._actuator = actuator

    def __init__(self, platform, color: AgentColor):    
        try:
            self.platform = platform
            self.color = color
            self.adjacency_matrix = platform.initial_adjacency_matrix.copy()
            self.position = None

            #TODO: set color dynamically
            if color == AgentColor.BLACK:
                self.orbit = BLACK_ORBIT
            elif color == AgentColor.YELLOW:
                self.orbit = YELLOW_ORBIT

            self.ref_trajectory = np.tile(self.orbit, NUM_SAMPLES)
            self.input_trajectory = self.ref_trajectory.copy()
            self.target_coil_idx = self.input_trajectory[0]

        except AttributeError:
            raise AttributeError(f"the {color} agent instance failed to initialize successfully.")
        
    def advance(self):
        i = self.platform.current_control_iteration

        if i < len(self.input_trajectory):
            error = np.linalg.norm(self.calc_raw_coordinates_by_idx(self.ref_trajectory[i]) - self.position)
            print("current_position: ", self.position, "ref position: ", self.ref_trajectory[i], "grid coordinates: ", self.calc_grid_coordinates(self.ref_trajectory[i]), "raw coordinates: ", self.calc_raw_coordinates_by_idx(self.ref_trajectory[i]))
            print("error", error)
            if error <= FIELD_RANGE:
                print("INSIDE field range of ref coil ...")
                self.__actuate(self.input_trajectory[i])
            else:
                print("OUTSIDE field range of ref coil...")
                closest_idx = self.find_closest_coil()
                self.__actuate(closest_idx)
                shortest_path = self.calculate_shortest_path(closest_idx)
                
                # self.update_input_plan([shortest_path[0], shortest_path[1]])
                self.input_trajectory[i] = shortest_path[0]
                self.input_trajectory[i+1] = shortest_path[1]
                self.__actuate(self.input_trajectory[i])
                self.__actuate(self.input_trajectory[i+1])

    def update_input_plan(self, inputs):
        for s, step in enumerate(inputs):
            input_step = self.platform.current_control_iteration + s
            self.input_trajectory[input_step] = inputs[s]
            
    def find_closest_coil(self):
        min_distance = math.inf
        closest_idx = 0
        for coil_idx in range(NUM_COILS):
            candidate_coordinates = self.calc_raw_coordinates_by_idx(coil_idx)
            distance = np.linalg.norm(self.position - candidate_coordinates)
            if distance < min_distance:
                min_distance = distance
                closest_idx = coil_idx

        return closest_idx
            
    def update_position(self, new_position):
        self.position = self.__coerce_position(new_position)

    def calculate_shortest_path(self, current_position_idx):
        graph = nx.from_numpy_array(self.adjacency_matrix)
        ref_position_idx = self.ref_trajectory[self.platform.current_control_iteration]
        return nx.dijkstra_path(graph, current_position_idx, ref_position_idx)
        
    def calc_grid_coordinates(self, idx):
        row = idx // GRID_WIDTH
        col = idx % GRID_WIDTH
        # print(f"calc grid coordinates: idx: {idx}, Row: {row}, Col: {col}")
        return row, col

    def calc_raw_coordinates_by_idx(self, idx):
        return self.calc_raw_coordinates_by_pos(*self.calc_grid_coordinates(idx))

    def calc_raw_coordinates_by_pos(self, row, col):
        return self.platform.grid_positions[row][col]

    def __actuate(self, idx):
        self._actuator.actuate_single(*self.calc_grid_coordinates(idx))
        
    def __coerce_position(self, measured_position):
        '''
            - coerce the current position to the raw coordinates of the nearest coil if within the coersion threshold
            - this helps filter for tracking noise and discretizes the measured position to that of the nearest coil
        '''

        target_coil_idx = self.input_trajectory[self.platform.current_control_iteration]
        coil_position = self.calc_raw_coordinates_by_idx(target_coil_idx)
        within_threshold = np.linalg.norm(coil_position - np.array(measured_position)) <= COERSION_THRESHOLD

        return coil_position if within_threshold else measured_position