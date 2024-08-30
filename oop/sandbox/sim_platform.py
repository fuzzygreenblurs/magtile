from sim_agent import SimAgent
import numpy as np
import pdb 
import asyncio
from shared_data import shared_data
import time


class Platform:
    GRID_WIDTH             = 15
    YELLOW_ORBIT           = [26, 27]
    BLACK_ORBIT            = [112, 112, 97, 97, 81, 81, 80, 80, 94, 94, 109, 109, 125, 125, 126, 126]
    # BLACK_ORBIT            = [112, 97, 81, 80, 94, 109, 125, 126]
    INTIAL_BLACK_POSITION  = [-4, -4]
    INTIAL_YELLOW_POSITION = [6, -4]
    NUM_SAMPLES            = 3200
    FIELD_RANGE            = 3.5

    def __init__(self):
        self.grid_x, self.grid_y = np.meshgrid(np.arange(self.GRID_WIDTH), np.arange(self.GRID_WIDTH))
        self.generate_adjacency_matrix()
        self.create_agents()

        self.black_agent = [a for a in self.agents if a.color == "black"][0]
        self.yellow_agent = [a for a in self.agents if a.color == "yellow"][0]

    def advance_agents(self):
        [a.advance() for a in self.agents]

    def update_all_agent_positions(self):
        if self.current_control_iteration == 0:
            return 
        
        # this simulates updating the stored position after reading from the prior actuation step
        for a in self.agents:
            # for the sake of simulation, we will assume readings come-in cartesian form
            a.position = a.position_at_end_of_prior_iteration
            print(f"{a.color}: {a.position}")

    def plan_for_interference(self):
        pass

    def generate_adjacency_matrix(self):
        num_coils = self.GRID_WIDTH * self.GRID_WIDTH
        grid_shape = (self.GRID_WIDTH, self.GRID_WIDTH)
        A = np.full((num_coils, num_coils), np.inf)

        for i in range(self.GRID_WIDTH):
            for j in range(self.GRID_WIDTH):
                current_idx = np.ravel_multi_index((i, j), grid_shape)
                neighbors = np.array([
                    [i, j - 1],
                    [i, j + 1],
                    [i - 1, j],
                    [i + 1, j],
                    [i - 1, j - 1],
                    [i - 1, j + 1],
                    [i + 1, j - 1],
                    [i + 1, j + 1],
                ])

                for n_i, n_j in neighbors:
                    if 0 <= n_i < self.GRID_WIDTH and 0 <= n_j < self.GRID_WIDTH:
                        neighbor_index = np.ravel_multi_index((int(n_i), int(n_j)), grid_shape)
                        distance = np.linalg.norm(np.array([i, j]) - np.array([n_i, n_j]))
                        A[current_idx, neighbor_index] = distance
                        A[neighbor_index, current_idx] = distance

        self.initial_adjacency_matrix = A

    def create_agents(self):
        black = SimAgent(self, "black", self.BLACK_ORBIT, self.INTIAL_BLACK_POSITION)
        # self.agents = [black]
        yellow = SimAgent(self, "yellow", self.YELLOW_ORBIT, self.INTIAL_YELLOW_POSITION)
        self.agents = [black, yellow]

    def idx_to_grid(self, idx):
        row = idx // self.GRID_WIDTH
        col = idx % self.GRID_WIDTH
        return row, col

    def grid_to_idx(self, row, col):
        return (row * self.GRID_WIDTH) + col

    def cartesian_to_idx(self, x, y):
        # Convert Cartesian coordinates to grid indices
        i = int(7 - y)  # Y-axis: Invert the y-coordinate and shift
        j = int(x + 7)  # X-axis: Shift the x-coordinate

        # Convert grid indices to scalar index
        index = np.int64(i * self.GRID_WIDTH + j)

        return index
    
    def grid_to_cartesian(self, x, y):
        # Convert coordinates from the upper left origin to the centered origin
        center_offset = (self.GRID_WIDTH - 1) // 2
        x_centered = y - center_offset
        y_centered = center_offset - x
        return x_centered, y_centered