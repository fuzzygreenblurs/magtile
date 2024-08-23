import constants
import numpy as np
import networkx as nx
import math
import threading
from constants import *
from agent_color import AgentColor

class Agent:
    _actuator = None

    @classmethod
    def set_actuator(cls, actuator):
        cls._actuator = actuator

    def __init__(self, color: AgentColor, adjacency_matrix):    
        try:
            self.color = color
            self.adjacency_matrix = adjacency_matrix
            self.position = None
            self.target_coil_idx = None
            self.orbit = getattr(constants, f"{color}_ORBIT")
            self.ref_trajectory = np.tile(self.target_path, NUM_SAMPLES)
            self.input_trajectory = self.ref_trajectory.copy()
        except AttributeError:
            raise AttributeError(f"the {color} agent instance failed to initialize successfully.")
        
    def advance(self, control_loop_increment):
        i = control_loop_increment

        if i < len(self.input_trajectory) - 1:
            error = np.linalg.norm(self.get_raw_coordinates(self.ref_trajectory[i]) - self.position)
            if error <= FIELD_RANGE:
                self.__actuate(self.input_trajectory[i])
            else:
                closest_idx, _ = self.find_closest_coil()
                self.__actuate(closest_idx)
                shortest_path = self.calculate_shortest_path(closest_idx)
                
                self.input_trajectory[i] = shortest_path[0]
                self.input_trajectory[i+1] = shortest_path[1]
                self.__actuate(self.input_trajectory[i])
                self.__actuate(self.input_trajectory[i+1])
            
    def update_position(self, new_position):
        self.position = self.__coerce_position(new_position)

    def find_closest_coil(self):
        min_distance = math.inf
        closest_idx = 0
        for i in range(self.grid_size * self.grid_size):
            candidate_coordinates = self.get_raw_coordinates(i)
            distance = np.linalg.norm(self.position - candidate_coordinates)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i

        closest_coil = self.calc_grid_coordinates(closest_idx)
        return closest_idx, closest_coil

    def calculate_shortest_path(self, current_position_index):
        graph = nx.from_numpy_array(self.adjacency_matrix)
        ref_position_idx = self.input_trajectory[self.current_index]
        return nx.dijkstra_path(graph, current_position_index, ref_position_idx)
    
    def __actuate(self, coil_index):
        self._actuator.actuate_single(*self.calc_grid_coordinates(coil_index))
        
    def __coerce_position(self, measured_position):
        '''
            - coerce the current position to the raw coordinates of the nearest coil if within the coersion threshold
            - this helps filter for tracking noise and discretizes the measured position to that of the nearest coil
        '''

        coil_position = self.calc_raw_coordinates(self.target_coil_idx)
        within_threshold = np.linalg.norm(coil_position - np.array(measured_position)) <= COERSION_THRESHOLD
        return coil_position if within_threshold else measured_position
