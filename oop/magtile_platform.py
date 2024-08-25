import pdb
import json
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
            if self.any_agents_inside_interference_zone():
                print("Agents inside interference zone. Recalculating paths.")
                self.recalculate_paths_for_agents()

            await self.advance_agents()

    async def advance_agents(self):
        await asyncio.gather(*[a.advance() for a in self.agents])

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
                agent.update_position(payload)
        else:
            return None

    def any_agents_inside_interference_zone(self):
        agent_positions = []
        for a in self.agents:
            if a.is_out_of_range():
                continue
            agent_positions.append(a.position)

        if len(agent_positions) == 1:
            return False

        for i in range(len(agent_positions)):
            for j in range(i + 1, len(agent_positions)):
                if np.linalg.norm(agent_positions[i] - agent_positions[j]) <= INTERFERENCE_RANGE:
                    return True
        return False

    ######### multi-agent Dijkstra ##########

    def recalculate_paths_for_agents(self):
        for i, agent in enumerate(self.agents):
            other_agents = [a for j, a in enumerate(self.agents) if j != i]
            shortest_safe_path = self.__calc_safe_path(agent, other_agents)
            agent.update_motion_plan(shortest_safe_path)

    def __calc_safe_path(self, agent, other_agents):
        adjusted_adjacency_matrix = self.initial_adjacency_matrix.copy()

        for other_agent in other_agents:
            adjusted_adjacency_matrix = self.__generate_safe_adjacency(
                adjusted_adjacency_matrix,
                other_agent.input_trajectory
            )

        return agent.multi_agent_shortest_path(adjusted_adjacency_matrix)

    def __generate_safe_adjacency(self, adjacency_matrix, other_agent_path):
        for i in range(len(other_agent_path)):
            position = np.array(Agent.calc_grid_coordinates(other_agent_path[i]))
            for j in range(len(adjacency_matrix)):
                distance = np.linalg.norm(position - np.array(Agent.calc_grid_coordinates(j)))

                if distance <= INTERFERENCE_RANGE:
                    adjacency_matrix[other_agent_path[i], j] = np.inf
                    adjacency_matrix[j, other_agent_path[i]] = np.inf

        return adjacency_matrix

    ######### multi-agent Dijkstra ##########

    ###### Initializer Functions ######

    def generate_meshgrid(self):
        x_lower = -(GRID_WIDTH - 1) / 2
        x_upper = (GRID_WIDTH - 1) / 2
        x_range = np.linspace(x_lower, x_upper, GRID_WIDTH) * COIL_SPACING

        y_lower = -(GRID_WIDTH - 1) / 2
        y_upper = (GRID_WIDTH - 1) / 2
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

    ###### Initializer Functions ######