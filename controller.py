import json
import numpy as np 
import networkx as nx
import math
import pdb
import time 
import serial
import socket
from actuator import Actuator

###################################
# Define Constants
###################################
ACTUATOR_PORT = "/dev/cu.usbmodem11301"
SOCKET_DOMAIN = "localhost"
SOCKET_PORT   = 65432

GRID_WIDTH            = 15                                                      # grid dimensions for static dipoles (x-direction)
FIELD_RANGE           = 3.048                                                   # magnetic force range: 3.048                           
COIL_SPACING          = 2.159                                                   # spacing between static dipoles: 2.159 (in cm)                

ref_trajectory_period = 200                                                     # total time period (s)
sampling_period       = 0.0625                                                  # camera sampling period
num_samples           = int(np.ceil(ref_trajectory_period / sampling_period))

################################
# Define Target Path By Position Index
################################
TARGET_PATH = [19, 20, 35, 49, 48, 33]

###########################################
# Generate Grid for Static Dipoles (Coils)
###########################################

x_lower = -(GRID_WIDTH - 1) / 2
x_upper = (GRID_WIDTH - 1) / 2
x_range = np.linspace(x_lower, x_upper, GRID_WIDTH ) * COIL_SPACING

y_lower = -(GRID_WIDTH - 1) / 2
y_upper = (GRID_WIDTH - 1) / 2
y_range = np.linspace(y_upper, y_lower, GRID_WIDTH ) * COIL_SPACING
# y_range = np.linspace(y_lower, y_upper, GRID_WIDTH ) * COIL_SPACING

x_grid, y_grid = np.meshgrid(x_range, y_range)

# generate a 2D grid, representing coil position coordinates
xy_grid_cells = [[None for _ in range(GRID_WIDTH )] for _ in range(GRID_WIDTH)]
for i in range(GRID_WIDTH):
    for j in range(GRID_WIDTH):
        xy_grid_cells[i][j] = np.array([x_grid[i, j], y_grid[i, j]])

###################################
# Create Adjacency Matrix A
###################################

# create a 225x225 grid to store the distance from each vertex to its neighbors
num_coils = GRID_WIDTH * GRID_WIDTH 
A = np.zeros((num_coils, num_coils))

for i in range(GRID_WIDTH):
    for j in range(GRID_WIDTH):
        # convert grid index to linear index
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
            if 0 <= n_i < GRID_WIDTH  and 0 <= n_j < GRID_WIDTH:
                neighbor_index = np.ravel_multi_index((n_i, n_j), x_grid.shape)
                distance = np.linalg.norm(np.array([i, j]) - np.array([n_i, n_j]))
                A[current_idx, neighbor_index] = distance
                A[neighbor_index, current_idx] = distance

###########################################
# Initialize Reference Signals
###########################################

def control(sock, actuator):
    
    ref_trajectory = np.tile(TARGET_PATH, num_samples)
    input_trajectory = ref_trajectory.copy()

    ###########################################
    # Control Loop
    ###########################################

    for i in range(num_samples):
        current_position = read_position(sock)
        ref_position_idx = ref_trajectory[i + 1]
        error = np.linalg.norm(get_raw_coordinates(ref_position_idx) - current_position)
        
        target_coil = calc_grid_coordinates(input_trajectory[i + 1])

        if error <= FIELD_RANGE:
            actuator.actuate_single(target_coil[0], target_coil[1])
        else:
            closest_idx, closest_coil = find_closest_coil(current_position)
            actuator.actuate_single(closest_coil[0], closest_coil[1])

            # compute shortest path to the reference trajectory: Djikstra
            graph = nx.from_numpy_array(A)
            shortest_path = nx.dijkstra_path(graph, closest_idx, ref_position_idx)
    
            # starting from the next iteration, follow the newly calculated shortest path:
            # this step replaces the current plan with the new shortest path plan until it reaches the pre-defined target coil
            input_trajectory[i + 1: ((i + 1) + len(shortest_path))] = shortest_path

            current_position = read_position(sock)
            
            # if the disk is within the radius of the second coil in the expected shortest path, skip directly to it
            if np.linalg.norm(current_position - xy_grid_cells[input_trajectory[1]]) <= FIELD_RANGE:
                target_coil_idx = input_trajectory[1]
                target_coil = np.unravel_index(target_coil_idx, x_grid.shape)
                actuator.actuate_single(target_coil[0], target_coil[1])
            else:
                target_coil_idx = input_trajectory[0]
                target_coil = np.unravel_index(target_coil_idx, x_grid.shape)
                actuator.actuate_single(target_coil[0], target_coil[1])

def control_test(sock):
    while True:
        data, _ = sock.recvfrom(1024)  # Buffer size is 1024 bytes
        payload = json.loads(data.decode())
        print("yellow x:", payload["yellow"][0], "yellow y:", payload["yellow"][1])


def read_position(sock):
    data, _ = sock.recvfrom(1024)  # Buffer size is 1024 bytes
    payload = json.loads(data.decode())
    # print("yellow x:", payload["yellow"][0], "yellow y:", payload["yellow"][1])
    return np.array(payload["yellow"])

def calc_grid_coordinates(index):
    return np.unravel_index(index, x_grid.shape)

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

    closest_coil = calc_grid_coordinates(closest_idx)
    return closest_idx, closest_coil

if __name__ == '__main__':
    server_address = (SOCKET_DOMAIN, SOCKET_PORT)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(server_address)

        try:
            with Actuator(ACTUATOR_PORT) as actuator:
                try:
                    # Read dimensions
                    width = actuator.read_width()
                    height = actuator.read_height()
                    print(f"Width: {width}, Height: {height}")

                    addresses = actuator.scan_addresses()
                    print(f"Addresses: {addresses}")

                    # Store configuration
                    actuator.store_config()

                    try:
                        control(sock, actuator)
                    except KeyboardInterrupt:
                        actuator.stop_all()

                except Exception as e:
                    print(f"An error occurred during operations: {e}")
        except serial.SerialException:
            print("Failed to connect to the Arduino.")






            


            
                
