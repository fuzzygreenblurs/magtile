import asyncio
import math
import numpy as np
import networkx as nx
import pdb
from agent_color import AgentColor
from constants import *

class Agent:
    _actuator = None

    @classmethod
    def set_actuator(cls, actuator):
        cls._actuator = actuator

    def __init__(self, platform, color: AgentColor):
        try:            
            self.platform = platform
            self.color = color
            if OPERATION_MODE == "LIVE":
                self.position = np.array([OUT_OF_RANGE, OUT_OF_RANGE])
            
            if color == AgentColor.BLACK:
                self.orbit = BLACK_ORBIT
                if OPERATION_MODE == "SIMULATION":
                    self.update_position(SIMULATED_INITIAL_BLACK_POSITION)

            elif color == AgentColor.YELLOW:
                self.orbit = YELLOW_ORBIT
                if OPERATION_MODE == "SIMULATION":
                    self.update_position(SIMULATED_INITIAL_YELLOW_POSITION)

            self.ref_trajectory = np.tile(self.orbit, NUM_SAMPLES)
            self.input_trajectory = self.ref_trajectory.copy()
            self.adjacency_matrix = self.platform.initial_adjacency_matrix.copy()
            self.shortest_path = None

            if OPERATION_MODE == "SIMULATION":
                self.simulated_position_at_end_of_prior_iteration = self.position
                
        except AttributeError:
            raise AttributeError(f"The {color} agent instance failed to initialize successfully.")

    async def advance(self):
        if self.is_undetected():
            return

        i = self.platform.current_control_iteration
        if i < len(self.input_trajectory):
            ref_position = self.platform.idx_to_grid(self.ref_trajectory[i])

            error = np.linalg.norm(np.array(self.position) - np.array(ref_position))
            if error <= FIELD_RANGE:
                await self.__actuate(self.input_trajectory[i])
            else:
                if(self.motion_plan_updated_at_platform_level == False ):
                    shortest_path = self.single_agent_shortest_path()
                    self.update_motion_plan(shortest_path[:2])

                input_1 = self.platform.idx_to_grid(self.input_trajectory[i])
                input_2 = self.platform.idx_to_grid(self.input_trajectory[i+1])
                # print(f"sequentially actuate {input_1} and THEN {input_2}")
                await self.__actuate(self.input_trajectory[i])
                await self.__actuate(self.input_trajectory[i+1])

    def update_motion_plan(self, inputs):
        i = self.platform.current_control_iteration
        for s, step in enumerate(inputs):
            input_step = self.platform.current_control_iteration + s
            self.input_trajectory[input_step] = inputs[s]

    def single_agent_shortest_path(self):
        position_idx = int(self.platform.grid_to_idx(*self.position))
        graph = nx.from_numpy_array(self.adjacency_matrix)
        ref_position_idx = self.ref_trajectory[self.platform.current_control_iteration]
        self.shortest_path = nx.dijkstra_path(graph, position_idx, ref_position_idx)
        return self.shortest_path
    
    def is_close_to_reference(self):
        i = self.platform.current_control_iteration
        ref_trajectory_position = np.array(self.platform.idx_to_grid(self.ref_trajectory[i]))
        
        error = np.linalg.norm(ref_trajectory_position - self.position)
        if error <= FIELD_RANGE:
            return True
        
        return False
    
    def deactivate_positions_within_radius(self, target_idx):
        target_position = self.platform.idx_to_grid(target_idx)
        for i in range(GRID_WIDTH):
            for j in range(GRID_WIDTH):
                candidate_position = np.array([i, j])
                candidate_idx = self.platform.grid_to_idx(*candidate_position)
                
                if np.linalg.norm(target_position - candidate_position) <= DEACTIVATION_RADIUS:
                    self.platform.deactivated_positions.append(candidate_position)
                    neighbors = np.array([
                        [i, j - 1],
                        [i, j + 1],
                        [i - 1, j],
                        [i + 1, j],
                        [i - 1, j - 1],
                        [i - 1, j + 1],
                        [i + 1, j - 1],
                        [i + 1, j + 1]
                    ])

                    for neighbor in neighbors:
                        ni, nj = neighbor
                        if 0 <= ni <= GRID_WIDTH and 0 <= nj <= GRID_WIDTH:
                            neighbor_idx = self.platform.grid_to_idx(ni, nj)
                            self.adjacency_matrix[candidate_idx, neighbor_idx] = INVALIDATED_NODE_WEIGHT
                            self.adjacency_matrix[neighbor_idx, candidate_idx] = INVALIDATED_NODE_WEIGHT


    
    # def set_deactivated_positions_surrounding_target(self, target_idx):
    #     neighbors = self.get_one_layer_neighbors(target_idx)
    #     self.platform.deactivated_positions.append(neighbors)
    
    #     for neighbor_idx in neighbors:
    #         self.adjacency_matrix[neighbor_idx, :] = INVALIDATED_NODE_WEIGHT
    #         self.adjacency_matrix[:, neighbor_idx] = INVALIDATED_NODE_WEIGHT

    def get_one_layer_neighbors(self, position_idx):
        """
        Returns the indices of the positions that are one layer (directly adjacent)
        around the given position in the grid.
        """
        neighbors = []
        row, col = self.platform.idx_to_grid(position_idx)

        # Loop through adjacent cells
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < GRID_WIDTH and 0 <= j < GRID_WIDTH:
                    neighbors.append(self.platform.grid_to_idx(i, j))

        return neighbors
    
    def is_undetected(self):
        return np.any(self.position == OUT_OF_RANGE)

    def update_position(self, new_position):
        self.position = self.__coerce_position(new_position)
        # print(f"{self.color}: new position: {self.position}, new_position: {self.platform.grid_to_idx(*self.position)}")

    async def __actuate(self, new_position_idx):
        if OPERATION_MODE == "LIVE":
            await self._actuator.actuate_single(*self.platform.idx_to_grid(new_position_idx))
        
        elif OPERATION_MODE == "SIMULATION":
            new_position = self.platform.idx_to_grid(new_position_idx)
            self.simulated_position_at_end_of_prior_iteration = self.platform.grid_to_cartesian(*new_position)
            # print(f"{self.color} ACTUATE: {self.simulated_position_at_end_of_prior_iteration}")

    # TODO: coerce to the closest grid position
    def __coerce_position(self, measured_position):
        """
        - Coerce the current position to the raw coordinates of the nearest coil if within the coersion threshold.
        - This helps filter for tracking noise and discretizes the measured position to that of the nearest coil.
        """

        #TODO: are we updating to the correct grid position from the cartesian readings?
        if np.any(measured_position == OUT_OF_RANGE):
            return np.array([OUT_OF_RANGE, OUT_OF_RANGE])
        
        raw_grid = self.platform.cartesian_to_grid(*np.array(measured_position))
        if OPERATION_MODE == "SIMULATION":
            return raw_grid
        else:
            distances = np.linalg.norm(self.platform.coil_positions - np.array(measured_position), axis=1)
            closest_idx = np.argmin(distances)
            return np.array(self.platform.idx_to_grid(closest_idx))