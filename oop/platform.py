import json
import time
import threading
import numpy as np
import networkx as nx
from agent import Agent
from agent_color import AgentColor
from constants import *
'''
all values are measured in centimeters [cm] unless inches [in] are explicitly specified in variable name
variable notation: "x" represents a value in centimeters. "x_inches" represents a value in inches
'''

class Platform:
    def __init__(self, ipc_client):
        self.ipc_client = ipc_client
        self.x_grid, self.y_grid = self.populate_meshgrid()
        self.grid_positions = self.populate_grid_positions()
        self.adjacency_matrix = self.initialize_adjacency_matrix()
        self.create_agents()

    def control(self):
        '''
            for each iteration of the control loop:
                - if all agents are outside the interference range with each other, follow the individual shortest path calculated for each agent
                - if any agent falls within the interference range of another agent, calculate a new shortest path without any interference
        '''

        for i in NUM_SAMPLES:
            self.update_all_agent_positions()
            if self.any_agents_inside_interference_zone():
                # TODO: shortest_paths = djikstra_2()
                # TODO: [agent.update_motion_plan({steps: 2, shortest_paths) for path in shortest_paths]
                pass

            # [agent.advance(i) for agent in self.agents]
            self.advance_agents(i)

    def advance_agents(self, i):
        threads = []
        for agent in self.agents:
            thread = threading.Thread(target=agent.advance, args=(i,))
            threads.append(thread)

        [thread.start() for thread in threads]
        [thread.join() for thread in threads]

    def create_agents(self):
        '''
            - placeholder function to instantiate the target agents
            - TODO: this should be replaced with a more generic add_agent function instead 
        '''
        yellow = Agent(AgentColor.YELLOW, self.adjacency_matrix.copy())
        black  = Agent(AgentColor.BLACK, self.adjacency_matrix.copy())
        self.agents = [yellow, black]

    def update_all_agent_positions(self):
        messages = self.ipc_client.xrevrange(POSITIONS_STREAM, count=1)
        
        if messages:
            _, message = messages[0]
            for agent in self.agents:
                color = f"b\'{agent.color}\'".encode('utf-8')
                agent.update_position(json.loads(message[color].decode()))

        else:
            return None

    def populate_meshgrid(self):
        x_lower = -(GRID_WIDTH - 1) / 2
        x_upper =  (GRID_WIDTH - 1) / 2
        x_range = np.linspace(x_lower, x_upper, GRID_WIDTH) * COIL_SPACING

        y_lower = -(GRID_WIDTH - 1) / 2
        y_upper =  (GRID_WIDTH - 1) / 2
        y_range = np.linspace(y_upper, y_lower, GRID_WIDTH) * COIL_SPACING
        
        return np.meshgrid(x_range, y_range)
    
    def populate_coil_grid_positions(self):
        '''
            - generates a 2D grid, representing coil position coordinates
            - each entry represents the (x,y) coordinates of a coil on the grid
        '''

        x_grid, y_grid = self.__populate_meshgrid()

        grid_positions = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_WIDTH)]
        for i in range(GRID_WIDTH):
            for j in range(GRID_WIDTH):
                grid_positions[i][j] = np.array([self.x_grid[i, j], self.y_grid[i, j]])

        return grid_positions
        
    def initialize_adjacency_matrix(self):
        '''
            - generates a matrix representation (called A here) of how far each coil in the platform is to its neighbors
            - A.shape: NUM_COILS x NUM_COILS (ex. a 15x15 coil platform will generate a 225x225 shaped A matrix)
            - each coil should have a total of 8 adjacent neighbors (including diagonals)
        '''

        num_coils = GRID_WIDTH * GRID_WIDTH
        A = np.zeros((num_coils, num_coils))
        # A = np.full((num_coils, num_coils), np.inf)

        for i in range(GRID_WIDTH):
            for j in range(GRID_WIDTH):
                current_idx = np.ravel_multi_index((i, j), self.x_grid.shape)
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
                        neighbor_index = np.ravel_multi_index((n_i, n_j), self.x_grid.shape)
                        distance = np.linalg.norm(np.array([i, j]) - np.array([n_i, n_j]))
                        A[current_idx, neighbor_index] = distance
                        A[neighbor_index, current_idx] = distance

        return A

    def update_adjacency_for_collision_avoidance(self, path, threshold):
        '''
        Modify the adjacency matrix to avoid paths that are too close to a given path.
        '''

        A = self.adjacency_matrix.copy()
        for i in range(len(path)):
            for j in range(len(A)):
                distance = np.linalg.norm(np.array(self.grid_positions[path[i]]) - np.array(self.grid_positions[j]))
                if distance < threshold:
                    A[path[i], j] = np.inf
                    A[j, path[i]] = np.inf
        return A

    def djikstra_2(self, start_idx, end_idx, adjacency_matrix=None):
        '''
            - for each agent, calculate the shortest path from its current position to its target
            - djikstra_2 should be able to handle k agents and enforce the interference zone restriction for all of them 
        '''
        if adjacency_matrix is None:
            adjacency_matrix = self.adjacency_matrix
        graph = nx.from_numpy_array(adjacency_matrix)
        shortest_path = nx.dijkstra_path(graph, start_idx, end_idx)
        return shortest_path

    def any_agents_inside_interference_zone(self):
        '''
            for an arbitrary number of agents, check if each possible pair of agents is within the interference range
        '''

        agent_positions = [a.position for a in self.agents]

        for i in range(len(agent_positions)):
            for j in range(i + 1, len(agent_positions)):
                if np.linalg.norm(agent_positions[i] - agent_positions[j]) <= FIELD_RANGE:
                    return True
        return False