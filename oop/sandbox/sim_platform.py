from sim_agent import SimAgent
import numpy as np
import pdb 
import asyncio
from shared_data import shared_data
import time


class Platform:
    GRID_WIDTH             = 15
    YELLOW_ORBIT           = [26, 27, 42, 41]
    BLACK_ORBIT            = [112, 112, 97, 97, 81, 81, 80, 80, 94, 94, 109, 109, 125, 125, 126, 126]
    # BLACK_ORBIT            = [112, 97, 81, 80, 94, 109, 125, 126]
    # INTIAL_BLACK_POSITION  = [-4, -4]
    # INTIAL_YELLOW_POSITION = [6, -4]
    INITIAL_BLACK_POSITION  = [-4, -4]
    INITIAL_YELLOW_POSITION = [6, -7]
    NUM_SAMPLES            = 3200
    FIELD_RANGE            = 2

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
        # A = np.u((num_coils, num_coils))
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

        row_start, col_start = [8, 10]
        row_end, col_end = [11, 13]
    
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

        # Deactivate the section in the 2D grid from (8, 10) to (11, 13)
        row_start, col_start = [8, 10]
        row_end, col_end = [11, 13]
    
        for i in range(row_start, row_end + 1):
            for j in range(col_start, col_end + 1):
                current_idx = np.ravel_multi_index((i, j), grid_shape)
                A[current_idx, :] = np.inf
                A[:, current_idx] = np.inf

        self.initial_adjacency_matrix = A

    def create_agents(self):
        black = SimAgent(self, "black", self.BLACK_ORBIT, self.INITIAL_BLACK_POSITION)
        yellow = SimAgent(self, "yellow", self.YELLOW_ORBIT, self.INITIAL_YELLOW_POSITION)
        self.agents = [black, yellow]

    def idx_to_grid(self, idx):
        row = idx // self.GRID_WIDTH
        col = idx % self.GRID_WIDTH
        return row, col

    def grid_to_idx(self, row, col):
        return (row * self.GRID_WIDTH) + col
    
    def cartesian_to_grid(self, x, y):
        x_upper_left = int(7 - y)
        y_upper_left = int(x + 7)
        return x_upper_left, y_upper_left

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
    
    ###### INTEFERENCE #########

    # def plan_for_interference(self):
    #     if all(a.is_close_to_reference() for a in self.agents):
    #         print("no calculation necessary...")
    #         return
    #     else:
    #         print("far away...")
        
    #     # i = self.current_control_iteration
        
    #     # for a in self.agents:
    #     #     a.adjacency_matrix = self.initial_adjacency_matrix
        
    #     primary, secondary = self.prioritized_agents()
    #     distance_between_agents = np.linalg.norm(primary.position - secondary.position)
    #     # print("closest idx", primary.find_closest_coil(), secondary.find_closest_coil())
    #     # primary_closest_idx = primary.find_closest_coil()
        
    #     if distance_between_agents <= INTERFERENCE_RANGE:
    #     #     print("\n----- BEGIN: INTEFERENCE SUBROUTINE ----")
    #     #     print(f"distance between agents: {distance_between_agents}")
    #         if self.agents_far_far:
    #             primary_shortest_path = primary.single_agent_shortest_path(primary_closest_idx)
    #             primary.update_motion_plan(primary_shortest_path)
    #             primary.motion_plan_updated_at_platform_level = True

                # distances = [
                #     norm(black.positon - yellow.position),
                #     norm(black.positon - yellow.shortest_path[i]),
                #     norm(black.positon - yellow.shortest_path[i+1)
                # ]

                # target_postion = min(distances)




        #         distances = [(distance_between_agents, primary_closest_idx)]
        #         if len(primary_shortest_path) == 2:
        #             primary_shortest_path_1_position = np.array(Agent.calc_raw_coordinates_by_idx(primary_shortest_path[1]))
        #             distance_primary_ref_1_to_secondary = np.linalg.norm(secondary.position - primary_shortest_path_1_position)
        #             distances.append((distance_primary_ref_1_to_secondary, self.find_closest_grid_position_idx(primary_shortest_path_1_position)))

        #         if len(primary_shortest_path) == 3:
        #             primary_shortest_path_2_position = np.array(Agent.calc_raw_coordinates_by_idx(primary_shortest_path[2]))
        #             distance_primary_ref_2_to_secondary = np.linalg.norm(secondary.position - primary_shortest_path_2_position)
        #             distances.append((distance_primary_ref_2_to_secondary, self.find_closest_grid_position_idx(primary_shortest_path_2_position)))

        #         _ , deactivated_epicenter_idx = min(distances)
        #         deactivated_epicenter = Agent.calc_raw_coordinates_by_idx(deactivated_epicenter_idx)
        #         print(f"primary shortest path: {primary_shortest_path}")
        #         if len(primary_shortest_path) == 2:
        #             primary_ref_position = np.array(Agent.calc_grid_coordinates(primary_shortest_path[i+1]))
        #         elif len(primary_shortest_path) == 1:
        #             primary_ref_position = np.array(Agent.calc_grid_coordinates(primary_shortest_path[i]))

        #     else:
        #         primary_ref_position = np.array(Agent.calc_grid_coordinates(primary.input_trajectory[i+1]))
        #         distance_between_secondary_and_primary_ref_pos = np.linalg.norm(secondary.position - primary_ref_position)
        #         deactivated_epicenter = primary.position if (distance_between_secondary_and_primary_ref_pos > distance_between_agents) else primary_ref_position
        #         deactivated_epicenter_idx = self.find_closest_grid_position_idx(deactivated_epicenter)

        #     # print("deacticated_epicenter_idx", deactivated_epicenter_idx)

        #     # deactivated_adjacency_matrix = self.updated_adjacency_matrix(self.find_closest_grid_position_idx(primary_closest_idx))
        #     deactivated_adjacency_matrix = self.updated_adjacency_matrix(deactivated_epicenter_idx)
        #     # deactivated_adjacency_matrix = self.updated_adjacency_matrix(primary_closest_idx)

        #     secondary.adjacency_matrix = deactivated_adjacency_matrix
        #     secondary_shortest_path = secondary.single_agent_shortest_path()
        #     secondary.update_motion_plan(secondary_shortest_path)
        #     secondary.motion_plan_updated_at_platform_level = True

        #     print(f"secondary shortest path: {secondary_shortest_path}")
        #     print("----- END: INTEFERENCE SUBROUTINE ---- \n")

    # def prioritized_agents(self):
    #     yellow = self.retrieve_agent(AgentColor.YELLOW)
    #     black  = self.retrieve_agent(AgentColor.BLACK)
    #     self.agents_far_far = False

    #     if yellow.is_close_to_reference() and not black.is_close_to_reference():
    #         print("yellow: close, black: far")
    #         return yellow, black
    #     elif black.is_close_to_reference() and not yellow.is_close_to_reference():
    #         print("black: close, yellow: far")
    #         return black, yellow
    #     else:
    #         print("yellow: far, black: far")
    #         self.agents_far_far = True
    #         return yellow, black

    # def updated_adjacency_matrix(self, target_idx):
    #     #     distances = [
    #         #     norm(black.positon - yellow.position),
    #         #     norm(black.positon - yellow.shortest_path[i]),
    #         #     norm(black.positon - yellow.shortest_path[i+1)
    #         # ]

    #     A = self.initial_adjacency_matrix.copy()

    #     # neighbors = self.get_three_layer_neighbors(target_idx)
    #     # neighbors = self.get_two_layer_neighbors(target_idx)
    #     neighbors = self.get_one_layer_neighbors(target_idx)
    
    #     for neighbor_idx in neighbors:
    #         A[neighbor_idx, :] = 1000
    #         A[:, neighbor_idx] = 1000

    #     return A
    
    # def get_one_layer_neighbors(self, position_idx):
    #     """
    #     Returns the indices of the positions that are one layer (directly adjacent)
    #     around the given position in the grid.
    #     """
    #     neighbors = []
    #     row, col = Agent.calc_grid_coordinates(position_idx)

    #     # Loop through adjacent cells
    #     for i in range(row - 1, row + 2):
    #         for j in range(col - 1, col + 2):
    #             if 0 <= i < GRID_WIDTH and 0 <= j < GRID_WIDTH:
    #                 if not (i == row and j == col):  # Exclude the center position
    #                     neighbors.append(self.calc_closest_idx(i, j))

    #     return neighbors

    # def get_two_layer_neighbors(self, position_idx):
    #     neighbors = []
    #     row, col = Agent.calc_grid_coordinates(position_idx)

    #     # First layer (directly adjacent)
    #     for i in range(row - 1, row + 2):
    #         for j in range(col - 1, col + 2):
    #             if 0 <= i < GRID_WIDTH and 0 <= j < GRID_WIDTH:
    #                 if not (i == row and j == col):  # Exclude the center position
    #                     neighbors.append(self.calc_closest_idx(i, j))

    #     # Second layer (one step further out)
    #     for i in range(row - 2, row + 3):
    #         for j in range(col - 2, col + 3):
    #             if 0 <= i < GRID_WIDTH and 0 <= j < GRID_WIDTH:
    #                 if abs(i - row) == 2 or abs(j - col) == 2:  # Ensure it's the second layer
    #                     neighbors.append(self.calc_closest_idx(i, j))

    #     return neighbors
    
    # def get_three_layer_neighbors(self, position_idx):
        # neighbors = set()
        # row, col = Agent.calc_grid_coordinates(position_idx)

        # # Add neighbors within 3 layers
        # for i in range(row - 3, row + 4):
        #     for j in range(col - 3, col + 4):
        #         if 0 <= i < GRID_WIDTH and 0 <= j < GRID_WIDTH:
        #             neighbors.add(self.calc_closest_idx(i, j))

        # # Exclude the yellow disc's current position
        # # neighbors.discard(position_idx)

        # return list(neighbors)

    ###### Initializer Functions ######