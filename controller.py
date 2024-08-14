import redis
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
ACTUATOR_PORT = "/dev/cu.usbmodem21301"
# SOCKET_DOMAIN = "localhost"
# SOCKET_PORT   = 65432
REDIS_CLIENT = redis.StrictRedis(host='localhost', port=6379, db=0)
CHANNEL = "positions"
PUBSUB = REDIS_CLIENT.pubsub()
PUBSUB.subscribe(CHANNEL)

r = redis.Redis(host='localhost', port=6379, db=0)
stream_name = 'stream_positions'


GRID_WIDTH            = 15                                                      # grid dimensions for static dipoles (x-direction)
# FIELD_RANGE           = 4                                                     # magnetic force range: 3.048                           
FIELD_RANGE           = 3.048                                                   # magnetic force range: 3.048                           
COIL_SPACING          = 2.159                                                   # spacing between static dipoles: 2.159 (in cm)                
COERSION_THRESHOLD    = 0.762                                                   # anything within 0.3 inches of the coil radius can be coerced to coil centroid position

REF_TRAJECTORY_PERIOD = 200                                                     # total time period (s)
SAMPLING_PERIOD       = 0.0625                                                  # camera sampling period
NUM_SAMPLES           = int(np.ceil(REF_TRAJECTORY_PERIOD / SAMPLING_PERIOD))
# TARGET_PATH           = [112, 97, 81, 80, 94, 109, 125, 126]

################################
# Define Target Path By Position Index
################################
TARGET_PATH = [112, 112, 97, 97, 81, 81, 80, 80, 94, 94, 109, 109, 125, 125, 126, 126]
# TARGET_PATH = [19, 20, 35, 49, 48, 33]

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

def wait_for_msg():
    while True:
        message = PUBSUB.get_message()
        if message and message['type'] == 'message':
            data = json.loads(message['data'])
            if data:
                return data['yellow']
        # time.sleep(0.1)  # Small delay to prevent CPU overuse

# Initialize a variable to store the last message ID
last_message_id = '0'

def read_latest_position():
    # Get the most recent entry in the stream
    messages = r.xrevrange(stream_name, count=1)
    
    if messages:
        # Unpack the message
        message_id, message = messages[0]
        
        # Decode and return the message content
        timestamp = message[b'timestamp'].decode()
        yellow_pos = json.loads(message[b'yellow'].decode())
        red_pos = json.loads(message[b'red'].decode())
        return {
            "timestamp": timestamp,
            "yellow": yellow_pos,
            "red": red_pos
        }
    else:
        return None
    

# OL_TARGET_PATH = [112, 97, 81, 80, 94, 109, 125, 126]    
# def open_loop(actuator):
#     ref_trajectory = np.tile(OL_TARGET_PATH, NUM_SAMPLES)
#     print("open loop control for ref trajectory: ", TEST_TARGET_PATH, "\n")
#     for element in ref_trajectory:
#         target_coil = calc_grid_coordinates(element)
#         actuator.actuate_single(target_coil[0], target_coil[1])
#         time.sleep(1)

def to_in(cm):
    return round(cm * 0.393701, 2)

def control(actuator):
    ref_trajectory = np.tile(TARGET_PATH, NUM_SAMPLES)
    input_trajectory = ref_trajectory.copy()

    ###########################################
    # Control Loop
    ###########################################
    # pdb.set_trace()

    for i in range(NUM_SAMPLES):
        print(f"\n--------------------")
        print(f"iteration: {i}")
        print(f"input_trajectory: ", input_trajectory[0:17])
        # print(f"input_trajectory: ", input_trajectory[i:i+17])
        ref_position_idx = input_trajectory[i]
        current_position = read_latest_position()["yellow"]

        if(i >= 1):
            current_position = coerce(input_trajectory[i-1], current_position)

        error = np.linalg.norm(get_raw_coordinates(ref_position_idx) - current_position)
        raw = get_raw_coordinates(ref_position_idx)
        print(f"ref_position_idx: {ref_position_idx}, cm: ({round(raw[0], 3)}, {round(raw[1], 3)}), in: ({to_in(raw[0]), to_in(raw[1])})")
        print(f"current position cm: ({current_position[0]}, {current_position[1]}), in: ({to_in(current_position[0])}, {to_in(current_position[1])})")

        if error <= FIELD_RANGE:
            target_coil = calc_grid_coordinates(ref_position_idx)
            raw = get_raw_coordinates(ref_position_idx)
            print(f"WITHIN range: target coil {ref_position_idx}", "cm: ", f"({raw[0]}, {raw[1]})",  "in: ", f"({to_in(raw[0]), to_in(raw[1])})")
            actuator.actuate_single(target_coil[0], target_coil[1])
            # time.sleep(1)
        else:
            closest_idx, closest_coil = find_closest_coil(current_position)
            raw = get_raw_coordinates(closest_idx)
            print(f"OUTSIDE range: closest coil: {closest_idx}", "cm: ", f"({raw[0]}, {raw[1]})", "in: ", f"({to_in(raw[0])}, {to_in(raw[1])})")
            actuator.actuate_single(closest_coil[0], closest_coil[1])
            # time.sleep(1)

            # compute shortest path to the reference trajectory: Djikstra
            graph = nx.from_numpy_array(A)
            shortest_path = nx.dijkstra_path(graph, closest_idx, ref_position_idx)[1:]
            current_position = read_latest_position()["yellow"]
            
            shortest_first = get_raw_coordinates(shortest_path[0])
            print("shortest path: ", shortest_path, "next position cm: ", f"({round(shortest_first[0], 2)}, {round(shortest_first[1], 2)})", f"in: ({int(to_in(shortest_first[0]))}, {int(to_in(shortest_first[1]))})")
    
            # starting from the next iteration, follow the newly calculated shortest path:
            # this step replaces the current plan with the new shortest path plan until it reaches the pre-defined target coil
            # if the disk is within the radius of the second coil in the expected shortest path, skip directly to it
            if len(shortest_path) == 1:
                input_trajectory[i+1] = shortest_path[0]
            elif np.linalg.norm(current_position - get_raw_coordinates(shortest_path[1])) <= FIELD_RANGE:
                shortest_path = shortest_path[1:]
                input_trajectory[i+1: (i+1) + len(shortest_path)] = shortest_path
                print("within field range of 1st index: ", shortest_path[1], "next position: ", get_raw_coordinates(shortest_path[0]))
            else:
                print("within field range of 0th index: ", shortest_path[0], "next position: ", get_raw_coordinates(shortest_path[0]))
                input_trajectory[i+1: (i+1) + len(shortest_path)] = shortest_path

        current_position = read_latest_position()["yellow"]
        print("updated input trajectory: ", input_trajectory[0:15])
        print("stop all coils at end of iteration...")
        actuator.stop_all()

def coerce(coil_idx, current_position):
    '''
    hacky solution: the position tracker has inherent error in the h_pos and v_pos.
    thus, the disc can move to where the target_coil is positioned but have a different
    measured position. 
    
    given the tight tolerance of the coil FIELD_RANGE:
        if the current_position is within a threshold radius of a target_coil, consider 
        the current_position to be at the (x, y) position of the centroid of the coil. 
    '''
    coil_position = get_raw_coordinates(coil_idx)
    if np.linalg.norm(coil_position - np.array(current_position)) <= COERSION_THRESHOLD:
        return coil_position
    else:
        return current_position

def calc_grid_coordinates(index):
    row = index // GRID_WIDTH
    col = index % GRID_WIDTH
    # print(f"calc grid coordinates: index: {index}, Row: {row}, Col: {col}")
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

    print(f"closest coil is: {round(min_distance, 2)} cm away")

    closest_coil = calc_grid_coordinates(closest_idx)
    return closest_idx, closest_coil

if __name__ == '__main__':
    with Actuator(ACTUATOR_PORT) as actuator:
        try:
            # Read dimensions
            width = actuator.read_width()
            height = actuator.read_height()
            addresses = actuator.scan_addresses()
            actuator.store_config()

            try:
                control(actuator)
                # open_loop(actuator)

            except KeyboardInterrupt:
                actuator.stop_all()

        except Exception as e:
            print(f"An error occurred during operations: {e}")
        except serial.SerialException:
            print("Failed to connect to the Arduino.")






            


            
                
