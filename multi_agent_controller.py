import redis
import cv2
import json
import numpy as np
import networkx as nx
import math
import time
import serial
from actuator import Actuator

###################################
# Define Constants
###################################
ACTUATOR_PORT = "/dev/cu.usbmodem21301"
REDIS_CLIENT = redis.StrictRedis(host='localhost', port=6379, db=0)
CHANNEL = "positions"
PUBSUB = REDIS_CLIENT.pubsub()
PUBSUB.subscribe(CHANNEL)

r = redis.Redis(host='localhost', port=6379, db=0)
stream_name = 'stream_positions'

GRID_WIDTH = 15                                    # grid dimensions for static dipoles
FIELD_RANGE = 3.1                                  # magnetic force range in cm
COIL_SPACING = 2.159                               # spacing between static dipoles: 2.159 cm
COERSION_THRESHOLD_IN = 0.4                        # in inches
COERSION_THRESHOLD = COERSION_THRESHOLD_IN * 2.54  # convert to cm

REF_TRAJECTORY_PERIOD = 200                        # total time period (s)
SAMPLING_PERIOD = 0.0625                           # camera sampling period
NUM_SAMPLES = int(np.ceil(REF_TRAJECTORY_PERIOD / SAMPLING_PERIOD))

TARGET_PATHS = [
    [112, 97, 81, 80, 94, 109, 125, 126],
    [117, 102, 88, 89, 105, 120, 134, 133]
]


###########################################
# Discs
###########################################

# class Disc():
#     def __init__(self, orbit, color):
#         self.ref_trajectory = None
#         self.input_trajectory = None
#         self.orbit = orbit
#         self.current_position = None
#         self.color = color

#     def set_ref_trajectory(self, ref_trajectory):
#         self.ref_trajectory = ref_trajectory

#     def set_ref_trajectory(self, ref_trajectory):
#         self.ref_trajectory = ref_trajectory



###########################################
# Generate Grid for Static Dipoles (Coils)
###########################################

x_lower = -(GRID_WIDTH - 1) / 2
x_upper = (GRID_WIDTH - 1) / 2
x_range = np.linspace(x_lower, x_upper, GRID_WIDTH) * COIL_SPACING

y_lower = -(GRID_WIDTH - 1) / 2
y_upper = (GRID_WIDTH - 1) / 2
y_range = np.linspace(y_upper, y_lower, GRID_WIDTH) * COIL_SPACING

x_grid, y_grid = np.meshgrid(x_range, y_range)

# Generate a 2D grid, representing coil position coordinates
xy_grid_cells = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_WIDTH)]
for i in range(GRID_WIDTH):
    for j in range(GRID_WIDTH):
        xy_grid_cells[i][j] = np.array([x_grid[i, j], y_grid[i, j]])

###################################
# Create Adjacency Matrix A
###################################

num_coils = GRID_WIDTH * GRID_WIDTH
A = np.zeros((num_coils, num_coils))

for i in range(GRID_WIDTH):
    for j in range(GRID_WIDTH):
        current_idx = np.ravel_multi_index((i, j), x_grid.shape)
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
                neighbor_index = np.ravel_multi_index((n_i, n_j), x_grid.shape)
                distance = np.linalg.norm(np.array([i, j]) - np.array([n_i, n_j]))
                A[current_idx, neighbor_index] = distance
                A[neighbor_index, current_idx] = distance

###########################################
# Initialize Reference Signals
###########################################

def update_trajectory_with_new_path(input_trajectory, closest_idx, ref_trajectory, current_positions, discs, iteration):
    new_paths = calculate_non_intersecting_paths(A, closest_idx, ref_trajectory[iteration:], current_positions, discs)

    if len(new_paths) > 1:
        input_trajectory[iteration:iteration + len(new_paths[0])] = new_paths[0]
    return input_trajectory

def calculate_non_intersecting_paths(A, start_indices, end_indices, current_positions, discs):
    paths = []
    for start_idx, end_idx in zip(start_indices, end_indices):
        A_updated = update_adjacency_for_collision_avoidance(A.copy(), paths, xy_grid_cells, 2*FIELD_RANGE)
        path = calc_shortest_path(A_updated, start_idx, end_idx)
        paths.append(path)
    return paths

def update_adjacency_for_collision_avoidance(A, paths, xy_grid_cells, threshold):
    for path in paths:
        for i in range(len(path)):
            for j in range(len(A)):
                distance = np.linalg.norm(np.array(xy_grid_cells[path[i]]) - np.array(xy_grid_cells[j]))
                if distance < threshold:
                    A[path[i], j] = np.inf
                    A[j, path[i]] = np.inf
    return A

def coerce(coil_idx, current_position):
    coil_position = get_raw_coordinates(coil_idx)
    if np.linalg.norm(coil_position - np.array(current_position)) <= COERSION_THRESHOLD:
        return coil_position
    else:
        return current_position

def calc_grid_coordinates(index):
    row = index // GRID_WIDTH
    col = index % GRID_WIDTH
    return row, col

def calc_raw_coordinates(row, col):
    return xy_grid_cells[row][col]

def get_raw_coordinates(index):
    grid_coordinates = calc_grid_coordinates(index)
    return calc_raw_coordinates(grid_coordinates[0], grid_coordinates[1])

def find_closest_coil(current_position):
    min_distance = math.inf
    closest_idx = 0
    for i in range(num_coils):
        candidate_coordinates = get_raw_coordinates(i)
        distance = np.linalg.norm(current_position - candidate_coordinates)
        if(distance < min_distance):
            min_distance = distance
            closest_idx = i

    print(f"closest coil is: {round(min_distance, 2)} cm")
    return closest_idx

def read_latest_position():
    messages = r.xrevrange(stream_name, count=1)
    
    if messages:
        _ , message = messages[0]
        yellow_pos = json.loads(message[b'yellow'].decode())
        black_pos = json.loads(message[b'black'].decode())
        return yellow_pos, black_pos
    else:
        return None
    
def outside_interference_range(current_positions, field_range):
    """
    Check if all discs are far enough from each other.
    
    :param current_positions: Dictionary of current positions of all discs.
    :param field_range: The minimum required distance to avoid interference.
    :return: True if all discs are far enough from each other, False otherwise.
    """
    disc_ids = list(current_positions.keys())
    for i in range(len(disc_ids)):
        for j in range(i + 1, len(disc_ids)):
            distance = np.linalg.norm(np.array(current_positions[disc_ids[i]]) - np.array(current_positions[disc_ids[j]]))
            if distance < field_range:
                return False
    return True

def calc_shortest_path(A, start_idx, end_idx):
    """
    Calculate the shortest path using Dijkstra's algorithm.
    
    :param A: Adjacency matrix representing the graph.
    :param start_idx: The starting index of the path.
    :param end_idx: The ending index of the path.
    :return: A list of indices representing the shortest path from start_idx to end_idx.
    """
    graph = nx.from_numpy_array(A)
    shortest_path = nx.dijkstra_path(graph, start_idx, end_idx)
    return shortest_path

def actuate_two_steps(actuator, input_trajectory, i):
    target_coil = calc_grid_coordinates(input_trajectory[i])
    actuator.actuate_single(*target_coil)
    
    target_coil = calc_grid_coordinates(input_trajectory[i+1])
    actuator.actuate_single(*target_coil)

def control(discs):
    for disc in discs:
        disc['ref_trajectory'] = np.tile(disc['orbit'], NUM_SAMPLES)
        disc['input_trajectory'] = disc['ref_trajectory'].copy()

    for i in range(NUM_SAMPLES):
        yellow_pos, black_pos = read_latest_position()
        discs[0]['current_position'] = coerce(discs[0]['input_trajectory'][i], yellow_pos)
        discs[1]['current_position'] = coerce(discs[1]['input_trajectory'][i], black_pos)

        current_positions = [d['current_position'] for d in discs]
        if outside_interference_range(current_positions, FIELD_RANGE):
            for disc in discs:
                actuate_two_steps(disc['input_trajectory'], i)

        else:
            for disc in discs:



                closest_idx, _ = find_closest_coil(current_positions[disc_id])

                input_trajectories[disc_id] = update_trajectory_with_new_path(
                    input_trajectories[disc_id], 
                    closest_idx, 
                    ref_trajectories[disc_id], 
                    current_positions, 
                    discs, 
                    i
                )

                actuate_two_steps(input_trajectories[disc_id], i)

        actuator.stop_all()
        time.sleep(SAMPLING_PERIOD)

if __name__ == '__main__':
    discs = [
        {
            'orbit': [112, 97, 81, 80, 94, 109, 125, 126],
            'ref_trajectory': None,
            'input_trajectory': None,
            'current_position': None,
            'color': "yellow"
        },
        {
            'orbit': [[117, 102, 88, 89, 105, 120, 134, 133]],
            'ref_trajectory': None,
            'input_trajectory': None,
            'current_position': None,
            'color': 'black'
        }
    ]

    with Actuator(ACTUATOR_PORT) as actuator:
        try:
            # Read dimensions
            width = actuator.read_width()
            height = actuator.read_height()
            addresses = actuator.scan_addresses()
            actuator.store_config()

            try:
                control(discs)

            except KeyboardInterrupt:
                actuator.stop_all()

        except Exception as e:
            print(f"An error occurred during operations: {e}")
        except serial.SerialException:
            print("Failed to connect to the Arduino.")