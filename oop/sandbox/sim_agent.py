import networkx as nx
import numpy as np
import asyncio
import time
import pdb

class SimAgent:
    def __init__(self, platform, color, orbit, position):
        self.platform = platform
        self.color = color
        self.orbit = orbit
        self.ref_trajectory = np.tile(orbit, platform.NUM_SAMPLES)
        self.input_trajectory = self.ref_trajectory.copy()
        self.position = position
        self.adjacency_matrix = self.platform.initial_adjacency_matrix
        self.position_at_end_of_prior_iteration = position
        self.shortest_path = None
        self.deactivated_positions = []

    def advance(self):
        i = self.platform.current_control_iteration
        if i < len(self.input_trajectory):
            ref_position = self.platform.idx_to_grid(self.ref_trajectory[i])

            error = np.linalg.norm(np.array(self.position) - np.array(ref_position))
            if error <= self.platform.FIELD_RANGE:
                self.__actuate(self.input_trajectory[i])
            else:
                if(self.motion_plan_updated_at_platform_level == False ):
                    shortest_path = self.single_agent_shortest_path()
                    self.update_motion_plan(shortest_path[:2])

                self.__actuate(self.input_trajectory[i])
                self.__actuate(self.input_trajectory[i+1])

    def update_motion_plan(self, inputs):
        i = self.platform.current_control_iteration
        for s, step in enumerate(inputs):
            input_step = self.platform.current_control_iteration + s
            self.input_trajectory[input_step] = inputs[s]

    def single_agent_shortest_path(self):
        position_idx = int(self.platform.cartesian_to_idx(*self.position))
        graph = nx.from_numpy_array(self.adjacency_matrix)
        ref_position_idx = self.ref_trajectory[self.platform.current_control_iteration]
        self.shortest_path = nx.dijkstra_path(graph, position_idx, ref_position_idx)
        return self.shortest_path

    def __actuate(self, new_position_idx):
        i = self.platform.current_control_iteration
        new_position = self.platform.idx_to_grid(new_position_idx)
        self.position_at_end_of_prior_iteration = self.platform.grid_to_cartesian(*new_position)
        
        #TODO: make these functions async again to simulate concurrency
        # time.sleep(0)

    ###### INTEFERENCE #########
    def is_close_to_reference(self):
        i = self.platform.current_control_iteration
        
        ref_trajectory_position = np.array(self.platform.idx_to_grid(self.ref_trajectory[i]))
        current_position = np.array(self.platform.cartesian_to_grid(*self.position))

        error = np.linalg.norm(ref_trajectory_position - current_position)
        if error <= self.platform.FIELD_RANGE:
            return True
        
        return False
    
    def set_deactivated_positions_surrounding_target(self, target_idx):
        neighbors = self.get_one_layer_neighbors(target_idx)
        self.platform.deactivated_neighbors.append(neighbors)
    
        for neighbor_idx in neighbors:
            self.adjacency_matrix[neighbor_idx, :] = np.inf
            self.adjacency_matrix[:, neighbor_idx] = np.inf
    
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
                if 0 <= i < self.platform.GRID_WIDTH and 0 <= j < self.platform.GRID_WIDTH:
                    if not (i == row and j == col):  # Exclude the center position
                        neighbors.append(self.platform.grid_to_idx(i, j))

        return neighbors