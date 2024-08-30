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

    def advance(self):
        i = self.platform.current_control_iteration
        if i < len(self.input_trajectory):
            ref_position = self.platform.idx_to_grid(self.ref_trajectory[i])

            error = np.linalg.norm(np.array(self.position) - np.array(ref_position))
            if error <= self.platform.FIELD_RANGE:
                self.__actuate(self.input_trajectory[i])
            else:
                shortest_path = self.single_agent_shortest_path()
                self.update_motion_plan(shortest_path[:2])

                # grid_0 = self.platform.idx_to_grid(shortest_path[0])
                # grid_1 = self.platform.idx_to_grid(shortest_path[1])
                # print("")

                # cartesian_0 =  self.platform.grid_to_cartesian(*grid_0)
                # cartesian_1 =  self.platform.grid_to_cartesian(*grid_1)

                # print("shortest_path: ", shortest_path) {
                # print("grid coordinates: ", grid_0, grid_1)
                # print("cartesian coordinates: ", cartesian_0, cartesian_1})
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
        return nx.dijkstra_path(graph, position_idx, ref_position_idx)

    def __actuate(self, new_position_idx):
        i = self.platform.current_control_iteration
        new_position = self.platform.idx_to_grid(new_position_idx)
        self.position_at_end_of_prior_iteration = self.platform.grid_to_cartesian(*new_position)
        
        #TODO: make these functions async again to simulate concurrency
        # time.sleep(0)