import json
import time
import numpy as np
import networkx as nx
from agent_color import AgentColor
from constants import *
'''
all values are measured in centimeters [cm] unless inches [in] are explicitly specified in variable name
variable notation: "x" represents a value in centimeters. "x_inches" represents a value in inches
'''

class Platform:
    _current_iteration = 0

    @classmethod
    def increment_current_iteration(cls, increment=1):
        cls._current_iteration += increment

    def __init__(self):
        self.grid_size = GRID_WIDTH
        self.field_range = FIELD_RANGE

        self.x_grid, self.y_grid = self.populate_meshgrid()
        self.grid_positions = self.populate_grid_positions()
        self.adjacency_matrix = self.initialize_adjacency_matrix()
        self.agents = []

    def update_agent_positions(self):
        messages = IPC_CLIENT.xrevrange(POSITIONS_STREAM, count=1)
        
        if messages:
            _, message = messages[0]
            for agent in self.agents:
                color = f"b\'{agent.color}\'".encode('utf-8')
                agent.update_position(json.loads(message[color].decode()))

        else:
            return None

    def calc_grid_coordinates(self, index):
        row = index // GRID_WIDTH
        col = index % GRID_WIDTH
        # print(f"calc grid coordinates: index: {index}, Row: {row}, Col: {col}")
        return row, col

    def calc_raw_coordinates(self, index):
        grid_coordinates = self.calc_grid_coordinates(index)
        return self.calc_raw_coordinates(grid_coordinates[0], grid_coordinates[1])
    
    def calc_raw_coordinates(self, row, col):
        return self.grid_positions[row][col]
    
    def get_current_iteration(self):
        return self._current_iteration

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

    def add_agent(self, agent):
        self.agents.append(agent)

    def update_adjacency_for_collision_avoidance(self, path, threshold):
        """
        Modify the adjacency matrix to avoid paths that are too close to a given path.
        """
        A = self.adjacency_matrix.copy()
        for i in range(len(path)):
            for j in range(len(A)):
                distance = np.linalg.norm(np.array(self.grid_positions[path[i]]) - np.array(self.grid_positions[j]))
                if distance < threshold:
                    A[path[i], j] = np.inf
                    A[j, path[i]] = np.inf
        return A

    def calc_shortest_path(self, start_idx, end_idx, adjacency_matrix=None):
        if adjacency_matrix is None:
            adjacency_matrix = self.adjacency_matrix
        graph = nx.from_numpy_array(adjacency_matrix)
        shortest_path = nx.dijkstra_path(graph, start_idx, end_idx)
        return shortest_path

    def no_interference(self, agent_positions):
        '''
            for an arbitrary number of agents, check if each possible pair of agents is within the interference range
        '''

        for i in range(len(agent_positions)):
            for j in range(i + 1, len(agent_positions)):
                if np.linalg.norm(agent_positions[i] - agent_positions[j]) <= FIELD_RANGE:
                    return False
        return True

    def control(self):
        '''
            - for each iteration of the control loop:
                - if all agents are outside the interference range with each other, follow the individual shortest path calculated for each agent
                - if any agent falls within the interference range of another agent, calculate a new shortest path without any interference
        '''


        #TODO: this should only run for the duration of the experiment set by num_samples
        while True:
            agent_positions = [agent.position for agent in self.agents]

            if self.no_interference(agent_positions):
                for agent in self.agents:
                    agent.advance()
            else:
                for agent in self.agents:
                    current_position_index = agent.find_closest_coil()[0]
                    shortest_path = self.calc_shortest_path(current_position_index, agent.input_trajectory[agent.current_index])
                    agent.input_trajectory[agent.current_index:agent.current_index + len(shortest_path)] = shortest_path
                    agent.advance()

            # Stop all coils to avoid interference
            self.actuator.stop_all()
            time.sleep(0.1)