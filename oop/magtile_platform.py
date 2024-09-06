import pdb
import asyncio
import json
import numpy as np
from agent import Agent
from agent_color import AgentColor
from constants import *

class Platform:
    def __init__(self, ipc_client): 
        self.ipc_client = ipc_client
        self.generate_meshgrids()
        self.generate_coil_positions()
        self.generate_adjacency_matrix()
        self.create_agents()
        self.current_control_iteration = 0
        self.deactivated_positions = []

    def reset_interference_parameters(self):
        for a in self.agents:
            a.motion_plan_updated_at_platform_level = False
            a.adjacency_matrix = self.initial_adjacency_matrix.copy()
        
        self.deactivated_positions = []

    async def advance_agents(self):
        await asyncio.gather(*[a.advance() for a in self.agents])

    def update_agent_positions(self):
        '''
            in cartesian units wrt a centered origin
        '''

        if OPERATION_MODE == "LIVE":
            messages = self.ipc_client.xrevrange(POSITIONS_STREAM, count=1)
            if messages:
                _, message = messages[0]
                for agent in self.agents:
                    color = f"{agent.color.value}".encode('utf-8')
                    payload = json.loads(message[color].decode())
                    agent.update_position(payload)

        elif OPERATION_MODE == "SIMULATION":
            self.simulate_position_readings()

    def create_agents(self):
        self.black_agent = Agent(self, AgentColor.BLACK)
        self.yellow_agent = Agent(self, AgentColor.YELLOW)
        self.agents = [self.black_agent, self.yellow_agent]

    ## multi-agent Dijkstra ##
    
    def plan_for_interference(self):
        if any(a.is_undetected() for a in self.agents):
            return
        
        if all(a.is_close_to_reference() for a in self.agents):
            return
        
        i = self.current_control_iteration

        primary, secondary = self.prioritized_agents()
        distance_between_agents = np.linalg.norm(primary.position - secondary.position) * COIL_SPACING

        if distance_between_agents <= INTERFERENCE_RANGE:
            secondary.deactivate_positions_within_radius(self.grid_to_idx(*primary.position))

            secondary_sp = secondary.single_agent_shortest_path()
            secondary.update_motion_plan(secondary_sp)
            secondary.motion_plan_updated_at_platform_level = True

    def prioritized_agents(self):
        self.agents_far_far = False

        if self.yellow_agent.is_close_to_reference() and not self.black_agent.is_close_to_reference():
            return self.yellow_agent, self.black_agent
        elif self.black_agent.is_close_to_reference() and not self.yellow_agent.is_close_to_reference():
            return self.black_agent, self.yellow_agent
        else:
            return self.yellow_agent, self.black_agent
    ## initializer functions ##
    def generate_meshgrids(self):
        self.grid_x, self.grid_y = np.meshgrid(np.arange(GRID_WIDTH), np.arange(GRID_WIDTH))

        x_lower, x_upper = -(GRID_WIDTH - 1) / 2, (GRID_WIDTH - 1) / 2
        x_range = np.linspace(x_lower, x_upper, GRID_WIDTH) * COIL_SPACING
        y_lower, y_upper = -(GRID_WIDTH - 1) / 2, (GRID_WIDTH - 1) / 2
        y_range = np.linspace(y_upper, y_lower, GRID_WIDTH) * COIL_SPACING
        self.cartesian_grid_x, self.cartesian_grid_y = np.meshgrid(x_range, y_range)
    
    def generate_coil_positions(self):
        """
        the coil_positions grid is only used to convert the raw visually sensed position 
        into the corresponding grid coordinate. all other operations (including actuation) should
        be performed in terms of grid coordinates.
        """
        coil_positions = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_WIDTH)]
        for i in range(GRID_WIDTH):
            for j in range(GRID_WIDTH):
                coil_positions[i][j] = np.array([self.cartesian_grid_x[i, j], self.cartesian_grid_y[i, j]])

        self.coil_positions = [pos for row in coil_positions for pos in row]

    def generate_adjacency_matrix(self):
        num_coils = GRID_WIDTH * GRID_WIDTH
        grid_shape = (GRID_WIDTH, GRID_WIDTH)
        A = np.full((num_coils, num_coils), INVALIDATED_NODE_WEIGHT)

        for i in range(GRID_WIDTH):
            for j in range(GRID_WIDTH):
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
                    if 0 <= n_i < GRID_WIDTH and 0 <= n_j < GRID_WIDTH:
                        neighbor_index = np.ravel_multi_index((int(n_i), int(n_j)), grid_shape)
                        distance = np.linalg.norm(np.array([i, j]) - np.array([n_i, n_j]))
                        A[current_idx, neighbor_index] = distance
                        A[neighbor_index, current_idx] = distance

                A[current_idx, current_idx] = 0

        self.initial_adjacency_matrix = A

    ## helper functions ###

    def idx_to_grid(self, idx):
        row = idx // GRID_WIDTH
        col = idx % GRID_WIDTH
        return row, col

    def grid_to_idx(self, row, col):
        return (row * GRID_WIDTH) + col
    
    def cartesian_to_grid(self, x, y):
        x_upper_left = int(7 - y)
        y_upper_left = int(x + 7)
        return np.array([x_upper_left, y_upper_left])
    
    def grid_to_cartesian(self, x, y):
        # Convert coordinates from the upper left origin to the centered origin

        center_offset = (GRID_WIDTH - 1) // 2
        x_centered = y - center_offset
        y_centered = center_offset - x
        return np.array([x_centered, y_centered])
    
    def simulate_position_readings(self):
        if self.current_control_iteration == 0:
            return 
        
        for a in self.agents:
            a.update_position(a.simulated_position_at_end_of_prior_iteration)