import time
import pdb
import json
import asyncio
import math
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
        # print("num samples: ", NUM_SAMPLES)
        for i in range(NUM_SAMPLES):
            self.current_control_iteration = i
            print(f"\n------ {i} ------")
            # print("iteration: ", self.current_control_iteration)

            self.update_all_agent_positions()
            self.plan_for_interference()
            await self.advance_agents()

            # time.sleep(1)

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

    def retrieve_agent(self, color: AgentColor):
        return [a for a in self.agents if a.color == color][0]

    def plan_for_interference(self):
        if all(a.is_close_to_reference() for a in self.agents):
            print("no calculation necessary...")
            return
        
        i = self.current_control_iteration
        
        for a in self.agents:
            a.adjacency_matrix = self.initial_adjacency_matrix
        
        primary, secondary = self.prioritized_agents()
        distance_between_agents = np.linalg.norm(primary.position - secondary.position)

        if distance_between_agents <= INTERFERENCE_RANGE:
            print("\n----- BEGIN: INTEFERENCE SUBROUTINE ----")
            print(f"distance between agents: {distance_between_agents}")
            if self.agents_far_far:
                primary_closest_idx = primary.find_closest_coil()
                primary_shortest_path = primary.single_agent_shortest_path(primary_closest_idx)
                primary.update_motion_plan(primary_shortest_path[:3])

                distances = [(distance_between_agents, primary_closest_idx)]
                if len(primary_shortest_path) == 2:
                    primary_shortest_path_1_position = np.array(Agent.calc_raw_coordinates_by_idx(primary_shortest_path[1]))
                    distance_primary_ref_1_to_secondary = np.linalg.norm(secondary.position - primary_shortest_path_1_position)
                    distances.append((distance_primary_ref_1_to_secondary, self.find_closest_grid_position_idx(primary_shortest_path_1_position)))

                if len(primary_shortest_path) >= 3:
                    primary_shortest_path_2_position = np.array(Agent.calc_raw_coordinates_by_idx(primary_shortest_path[2]))
                    distance_primary_ref_2_to_secondary = np.linalg.norm(secondary.position - primary_shortest_path_2_position)
                    distances.append((distance_primary_ref_2_to_secondary, self.find_closest_grid_position_idx(primary_shortest_path_2_position)))

                _ , deactivated_epicenter_idx = min(distances)
                deactivated_epicenter = Agent.calc_raw_coordinates_by_idx(deactivated_epicenter_idx)
                print(f"primary shortest path: {primary_shortest_path}")

            else:
                primary_ref_position = np.array(Agent.calc_grid_coordinates(primary.ref_trajectory[i]))
                distance_between_secondary_and_primary_ref_pos = np.linalg.norm(secondary.position - primary_ref_position)
                deactivated_epicenter = primary.position if (distance_between_secondary_and_primary_ref_pos > distance_between_agents) else primary_ref_position
                deactivated_epicenter_idx = self.find_closest_grid_position_idx(deactivated_epicenter)

            deactivated_adjacency_matrix = self.updated_adjacency_matrix(deactivated_epicenter_idx)
            # deactivated_adjacency_matrix = self.updated_adjacency_matrix(primary_closest_idx)

            secondary.adjacency_matrix = deactivated_adjacency_matrix
            secondary_shortest_path = secondary.single_agent_shortest_path()
            secondary.update_motion_plan(secondary_shortest_path[:2])

            print(f"secondary shortest path: {secondary_shortest_path}")
            print("----- END: INTEFERENCE SUBROUTINE ---- \n")

    def prioritized_agents(self):
        yellow = self.retrieve_agent(AgentColor.YELLOW)
        black  = self.retrieve_agent(AgentColor.BLACK)
        self.agents_far_far = False

        if yellow.is_close_to_reference() and not black.is_close_to_reference():
            print("yellow: close, black: far")
            return yellow, black
        elif black.is_close_to_reference() and not yellow.is_close_to_reference():
            print("black: close, yellow: far")
            return black, yellow
        else:
            print("yellow: far, black: far")
            self.agents_far_far = True
            return yellow, black

    def updated_adjacency_matrix(self, target_idx):
        A = self.initial_adjacency_matrix.copy()

        # neighbors = self.get_three_layer_neighbors(target_idx)
        # neighbors = self.get_two_layer_neighbors(target_idx)
        neighbors = self.get_one_layer_neighbors(target_idx)
    
        for neighbor_idx in neighbors:
            A[neighbor_idx, :] = np.inf
            A[:, neighbor_idx] = np.inf

        return A
    
    def get_one_layer_neighbors(self, position_idx):
        """
        Returns the indices of the positions that are one layer (directly adjacent)
        around the given position in the grid.
        """
        neighbors = []
        row, col = Agent.calc_grid_coordinates(position_idx)

        # Loop through adjacent cells
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < GRID_WIDTH and 0 <= j < GRID_WIDTH:
                    if not (i == row and j == col):  # Exclude the center position
                        neighbors.append(self.calc_closest_idx(i, j))

        return neighbors

    def get_two_layer_neighbors(self, position_idx):
        neighbors = []
        row, col = Agent.calc_grid_coordinates(position_idx)

        # First layer (directly adjacent)
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < GRID_WIDTH and 0 <= j < GRID_WIDTH:
                    if not (i == row and j == col):  # Exclude the center position
                        neighbors.append(self.calc_closest_idx(i, j))

        # Second layer (one step further out)
        for i in range(row - 2, row + 3):
            for j in range(col - 2, col + 3):
                if 0 <= i < GRID_WIDTH and 0 <= j < GRID_WIDTH:
                    if abs(i - row) == 2 or abs(j - col) == 2:  # Ensure it's the second layer
                        neighbors.append(self.calc_closest_idx(i, j))

        return neighbors
    
    def get_three_layer_neighbors(self, position_idx):
        neighbors = set()
        row, col = Agent.calc_grid_coordinates(position_idx)

        # Add neighbors within 3 layers
        for i in range(row - 3, row + 4):
            for j in range(col - 3, col + 4):
                if 0 <= i < GRID_WIDTH and 0 <= j < GRID_WIDTH:
                    neighbors.add(self.calc_closest_idx(i, j))

        # Exclude the yellow disc's current position
        # neighbors.discard(position_idx)

        return list(neighbors)

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
        A = np.full((num_coils, num_coils), np.inf)

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

    def find_closest_grid_position_idx(self, target_position):
        if target_position is None:
            raise ValueError

        min_distance = math.inf
        closest_idx = 0
        for coil_idx in range(NUM_COILS):
            candidate_coordinates = Agent.calc_raw_coordinates_by_idx(coil_idx)
            distance = np.linalg.norm(np.array(target_position) - candidate_coordinates)
            if distance < min_distance:
                min_distance = distance
                closest_idx = coil_idx

        return closest_idx

    def calc_closest_idx(self, row, col):
        return (row * GRID_WIDTH) + col