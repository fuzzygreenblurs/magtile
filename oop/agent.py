import math
import numpy as np
import networkx as nx
from agent_color import AgentColor
from constants import *
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
            self.position = [OUT_OF_RANGE, OUT_OF_RANGE]
            self.grid_position = [OUT_OF_RANGE, OUT_OF_RANGE]

            if color == AgentColor.BLACK:
                self.orbit = BLACK_ORBIT
            elif color == AgentColor.YELLOW:
                self.orbit = YELLOW_ORBIT

            self.ref_trajectory = np.tile(self.orbit, NUM_SAMPLES)
            self.input_trajectory = self.ref_trajectory.copy()
            self.target_coil_idx = self.input_trajectory[0]

        except AttributeError:
            raise AttributeError(f"The {color} agent instance failed to initialize successfully.")

    async def advance(self):
        if self.is_out_of_range():
            return

        i = self.platform.current_control_iteration
        if i < len(self.input_trajectory):
            error = np.linalg.norm(self.calc_raw_coordinates_by_idx(self.ref_trajectory[i]) - self.position)
            print("ref position: ", self.ref_trajectory[i], "grid coordinates: ", self.calc_grid_coordinates(self.ref_trajectory[i]), "raw coordinates: ", self.calc_raw_coordinates_by_idx(self.ref_trajectory[i]))
            print("current_position: ", self.position)
            print("error", error)
            if error <= FIELD_RANGE:
                print("INSIDE field range of ref coil ...")
                await self.__actuate(self.input_trajectory[i])
            else:
                print("OUTSIDE field range of ref coil...")
                closest_idx = self.find_closest_coil()
                shortest_path = self.single_agent_shortest_path(closest_idx)
                self.update_motion_plan(shortest_path[:2])
                await self.__actuate(self.input_trajectory[i])
                await self.__actuate(self.input_trajectory[i+1])

    def is_out_of_range(self):
        return np.array_equal(self.position, [OUT_OF_RANGE, OUT_OF_RANGE])

    def update_motion_plan(self, inputs):
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
        self.grid_position = self.find_closest_coil()

    def single_agent_shortest_path(self, current_position_idx):
        graph = nx.from_numpy_array(self.adjacency_matrix)
        ref_position_idx = self.ref_trajectory[self.platform.current_control_iteration]
        return nx.dijkstra_path(graph, current_position_idx, ref_position_idx)

    def multi_agent_shortest_path(self, adjacency_matrix=None):
        if adjacency_matrix is None:
            adjacency_matrix = self.adjacency_matrix

        graph = nx.from_numpy_array(adjacency_matrix)
        current_idx = self.find_closest_coil()
        # current_idx = self.calc_closest_idx(*self.calc_grid_coordinates(self.position))
        ref_position_idx = self.ref_trajectory[self.platform.current_control_iteration]
        shortest_path = nx.dijkstra_path(graph, current_idx, ref_position_idx)
        return shortest_path
    
    @classmethod
    def calc_grid_coordinates(cls, idx):
        row = idx // GRID_WIDTH
        col = idx % GRID_WIDTH
        return row, col

    def calc_raw_coordinates_by_idx(self, idx):
        return self.calc_raw_coordinates_by_pos(*self.calc_grid_coordinates(idx))

    def calc_raw_coordinates_by_pos(self, row, col):
        return self.platform.grid_positions[row][col]

    def calc_closest_idx(self, row, col):
        return (row * GRID_WIDTH) + col

    async def __actuate(self, idx):
        await self._actuator.actuate_single(*self.calc_grid_coordinates(idx))

    def __coerce_position(self, measured_position):
        """
        - Coerce the current position to the raw coordinates of the nearest coil if within the coersion threshold.
        - This helps filter for tracking noise and discretizes the measured position to that of the nearest coil.
        """
        self.target_coil_idx = self.input_trajectory[self.platform.current_control_iteration]
        coil_position = self.calc_raw_coordinates_by_idx(self.target_coil_idx)
        within_threshold = np.linalg.norm(coil_position - np.array(measured_position)) <= COERSION_THRESHOLD

        return np.array(coil_position) if within_threshold else np.array(measured_position)