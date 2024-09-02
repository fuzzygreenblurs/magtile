import asyncio
import json
import numpy as np
import pdb
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

    def reset_agent_flags(self):
        for a in self.agents:
            a.motion_plan_updated_at_platform_level = False 

    async def advance_agents(self):
        print("gets to advance agents in platform")
        await asyncio.gather(*[a.advance() for a in self.agents])

    def update_agent_positions(self):
        messages = self.ipc_client.xrevrange(POSITIONS_STREAM, count=1)
        if messages:
            _, message = messages[0]
            for agent in self.agents:
                color = f"{agent.color.value}".encode('utf-8')
                payload = json.loads(message[color].decode())
                agent.update_position(payload)
        else:
            return None

    def create_agents(self):
        self.black_agent = Agent(self, AgentColor.BLACK)
        self.yellow_agent = Agent(self, AgentColor.YELLOW)
        self.agents = [self.black_agent, self.yellow_agent]

    ## multi-agent Dijkstra ##
    
    def plan_for_interference(self):
        if any(a.is_undetected() for a in self.agents):
            return
        
        if all(a.is_close_to_reference() for a in self.agents):
            print("no calculation necessary...")
            return
        
        i = self.current_control_iteration
        
        for a in self.agents:
            a.adjacency_matrix = self.initial_adjacency_matrix.copy()
        
        primary, secondary = self.prioritized_agents()
        distance_between_agents = np.linalg.norm(primary.position - secondary.position)

        if distance_between_agents <= INTERFERENCE_RANGE:
            print("WITHIN INTERFERENCE RANGE...")
            primary_sp = primary.single_agent_shortest_path()
            primary.update_motion_plan(primary_sp)
            primary.motion_plan_updated_at_platform_level = True


            if self.agents_far_far:
                primary_projected_positions = [primary_sp[0], primary_sp[1], primary_sp[2]]
            else:
                primary_projected_positions = [primary_sp[0], primary_sp[1]]
            
                
            for position in primary_projected_positions:
                secondary.set_deactivated_positions_surrounding_target(position)

            secondary_sp = secondary.single_agent_shortest_path()
            secondary.update_motion_plan(secondary_sp)
            secondary.motion_plan_updated_at_platform_level = True

            print("gets to end of interference planning")

    def prioritized_agents(self):
        self.agents_far_far = False

        if self.yellow_agent.is_close_to_reference() and not self.black_agent.is_close_to_reference():
            print("yellow: close, black: far")
            return self.yellow_agent, self.black_agent
        elif self.black_agent.is_close_to_reference() and not self.yellow_agent.is_close_to_reference():
            print("black: close, yellow: far")
            return self.black_agent, self.yellow_agent
        else:
            print("yellow: far, black: far")
            self.agents_far_far = True
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