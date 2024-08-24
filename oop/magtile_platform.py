import pdb
import json
import threading
import time
import asyncio
import numpy as np
import networkx as nx
from agent import Agent
from agent_color import AgentColor
from constants import *

class Platform:
    def __init__(self, ipc_client):
        self.ipc_client = ipc_client
        self.generate_meshgrid()
        self.generate_grid_positions()
        self.generate_initial_adjacency_matrix()
        self.create_agents()

    async def control(self):
        print("num samples: ", NUM_SAMPLES)
        for i in range(NUM_SAMPLES):
            self.current_control_iteration = i
            print("\n------")
            print("iteration: ", self.current_control_iteration)    

            self.update_all_agent_positions()
            # if self.any_agents_inside_interference_zone():
            #     #TODO: djikstra2
            #     #TODO: update input trajectories for all agents
            #     pass
            
            await self.advance_agents()

            # time.sleep(1)

    async def advance_agents(self):
        # yellow_agent = [a for a in self.agents if a.color == AgentColor.YELLOW][0]
        # await asyncio.gather(yellow_agent.advance())
                             
        await asyncio.gather(*[a.advance() for a in self.agents])

    # def advance_agents(self, i):
    #     #TODO: switch to async processing
    #     # [a.advance() for a in self.agents]

    #     yellow_agent = [a for a in self.agents if a.color == AgentColor.YELLOW][0]
    #     yellow_agent.advance()

        # threads = []
        # for agent in self.agents:
        #     thread = threading.Thread(target=agent.advance, args=())
        #     threads.append(thread)

        # [thread.start() for thread in threads]
        # [thread.join() for thread in threads]

    def create_agents(self):
        yellow = Agent(self, AgentColor.YELLOW)
        black = Agent(self, AgentColor.BLACK)
        self.agents = [yellow, black]

    def update_all_agent_positions(self):
        messages = self.ipc_client.xrevrange(POSITIONS_STREAM, count=1)
        if messages:
            _, message = messages[0]
            for agent in self.agents:
                color = f"{agent.color.value}".encode('utf-8')
                payload = json.loads(message[color].decode())
                # if agent.color == AgentColor.YELLOW:
                #     print(f"REDIS READ: {agent.color} payload: {payload}")
                agent.update_position(payload)
        else:
            return None

    def update_adjacency_for_collision_avoidance(self, path, threshold):
        A = self.adjacency_matrix.copy()
        for i in range(len(path)):
            for j in range(len(A)):
                distance = np.linalg.norm(np.array(self.grid_positions[path[i]]) - np.array(self.grid_positions[j]))
                if distance < threshold:
                    A[path[i], j] = np.inf
                    A[j, path[i]] = np.inf
        return A

    def multi_agent_dijkstra(self, start_idx, end_idx, adjacency_matrix=None):
        if adjacency_matrix is None:
            adjacency_matrix = self.adjacency_matrix
        graph = nx.from_numpy_array(adjacency_matrix)
        shortest_path = nx.dijkstra_path(graph, start_idx, end_idx)
        return shortest_path

    def any_agents_inside_interference_zone(self):
        agent_positions = [a.position for a in self.agents]
        for i in range(len(agent_positions)):
            for j in range(i + 1, len(agent_positions)):
                if np.linalg.norm(agent_positions[i] - agent_positions[j]) <= FIELD_RANGE:
                    return True
        return False


    def generate_meshgrid(self):
        x_lower = -(GRID_WIDTH - 1) / 2
        x_upper =  (GRID_WIDTH - 1) / 2
        x_range = np.linspace(x_lower, x_upper, GRID_WIDTH) * COIL_SPACING

        y_lower = -(GRID_WIDTH - 1) / 2
        y_upper =  (GRID_WIDTH - 1) / 2
        y_range = np.linspace(y_upper, y_lower, GRID_WIDTH) * COIL_SPACING
        
        self.x_grid, self.y_grid = np.meshgrid(x_range, y_range)

    def generate_grid_positions(self):
        grid_positions = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_WIDTH)]
        for i in range(GRID_WIDTH):
            for j in range(GRID_WIDTH):
                grid_positions[i][j] = np.array([self.x_grid[i, j], self.y_grid[i, j]])

        self.grid_positions = grid_positions

    def generate_initial_adjacency_matrix(self):
        num_coils = GRID_WIDTH * GRID_WIDTH
        A = np.zeros((num_coils, num_coils))

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
                        neighbor_index = np.ravel_multi_index((int(n_i), int(n_j)), self.x_grid.shape)
                        distance = np.linalg.norm(np.array([i, j]) - np.array([n_i, n_j]))
                        A[current_idx, neighbor_index] = distance
                        A[neighbor_index, current_idx] = distance

        self.initial_adjacency_matrix = A