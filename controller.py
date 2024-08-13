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
ACTUATOR_PORT = "/dev/cu.usbmodem11301"
# SOCKET_DOMAIN = "localhost"
# SOCKET_PORT   = 65432
REDIS_CLIENT = redis.StrictRedis(host='localhost', port=6379, db=0)
CHANNEL = "positions"
PUBSUB = REDIS_CLIENT.pubsub()
PUBSUB.subscribe(CHANNEL)

r = redis.Redis(host='localhost', port=6379, db=0)
stream_name = 'stream_positions'


GRID_WIDTH            = 15                                                      # grid dimensions for static dipoles (x-direction)
FIELD_RANGE           = 3.048                                                   # magnetic force range: 3.048                           
COIL_SPACING          = 2.159                                                   # spacing between static dipoles: 2.159 (in cm)                

REF_TRAJECTORY_PERIOD = 200                                                     # total time period (s)
SAMPLING_PERIOD       = 0.0625                                                  # camera sampling period
NUM_SAMPLES           = int(np.ceil(REF_TRAJECTORY_PERIOD / SAMPLING_PERIOD))

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

def wait_for_msg():
    while True:
        message = PUBSUB.get_message()
        if message and message['type'] == 'message':
            data = json.loads(message['data'])
            if data:
                return data['yellow']
        time.sleep(0.1)  # Small delay to prevent CPU overuse

# Initialize a variable to store the last message ID
last_message_id = '0'

def get_latest_data():
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


def osc_test(actuator):
    while True:
        print("performing osc test") 

        print("----- expected: 0, 0 --------")
        actuator.actuate_single(7, 7)
        time.sleep(0.5)  # Ensure time for message to be published
        latest_data = get_latest_data()
        print(latest_data)

        actuator.stop_all()
        time.sleep(0.5)

        print("----- expected: 0, 2.6 --------")
        actuator.actuate_single(6, 7)
        time.sleep(0.5)  # Ensure time for message to be published
        latest_data = get_latest_data()
        print(latest_data)

        actuator.stop_all()
        time.sleep(0.5)



def control(sock, actuator):
    ref_trajectory = np.tile(TARGET_PATH, NUM_SAMPLES)
    input_trajectory = ref_trajectory.copy()

    ###########################################
    # Control Loop
    ###########################################

    for i in range(NUM_SAMPLES):
        print(f"\n--------------------")
        current_position = read_position(sock)
        print(f"iteration: {i}")
        print("input_trajectory: ", input_trajectory[0:15])
        print("current_position: ", current_position)
        ref_position_idx = input_trajectory[i]
        error = np.linalg.norm(get_raw_coordinates(ref_position_idx) - current_position)

        if error <= FIELD_RANGE:
            target_coil = calc_grid_coordinates(ref_position_idx)
            print(f"error within field range. actuate target coil {ref_position_idx} ({target_coil[0]}, {target_coil[1]})")
            actuator.actuate_single(target_coil[0], target_coil[1])
            time.sleep(SAMPLING_PERIOD)
        else:
            closest_idx, closest_coil = find_closest_coil(current_position)
            print(f"error outside field range. actuate closest coil {closest_idx} ({closest_coil[0]}, {closest_coil[1]})")
            actuator.actuate_single(closest_coil[0], closest_coil[1])
            time.sleep(SAMPLING_PERIOD)

            # compute shortest path to the reference trajectory: Djikstra
            graph = nx.from_numpy_array(A)
            shortest_path = nx.dijkstra_path(graph, closest_idx, ref_position_idx)
            current_position = read_position(sock)
            
            print("shortest path: ", shortest_path, "first coil: ", shortest_path[0], "target position: ", get_raw_coordinates(shortest_path[0]))
    
            # starting from the next iteration, follow the newly calculated shortest path:
            # this step replaces the current plan with the new shortest path plan until it reaches the pre-defined target coil
            # if the disk is within the radius of the second coil in the expected shortest path, skip directly to it
            if np.linalg.norm(current_position - get_raw_coordinates(shortest_path[1])) <= FIELD_RANGE:
                shortest_path = shortest_path[1:]
                input_trajectory[i+1: (i+1) + len(shortest_path)] = shortest_path
                print("within field range of 2nd index: ", input_trajectory[1], "target position: ", get_raw_coordinates(input_trajectory[1]))
            else:
                print("not within field range of 2nd index: ", input_trajectory[1], "target position: ", get_raw_coordinates(input_trajectory[1]))
                input_trajectory[i+1: (i+1) + len(shortest_path)] = shortest_path

        current_position = read_position(sock)
        print("current position: ", current_position)
        print("updated input trajectory: ", input_trajectory[0:15])
        print("stop all coils at end of iteration...")
        actuator.stop_all()
        # time.sleep(2)


def read_position(sock):
    data, address = sock.recvfrom(1024)
    payload = json.loads(data.decode())
    print(f"Received {payload["yellow"]} from {address}")

    # print("yellow x:", payload["yellow"][0], "yellow y:", payload["yellow"][1])
    return payload["yellow"]

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
    with Actuator(ACTUATOR_PORT) as actuator:
        try:
            # Read dimensions
            width = actuator.read_width()
            height = actuator.read_height()
            addresses = actuator.scan_addresses()
            actuator.store_config()

            try:
                # test_read_position(sock)
                osc_test(actuator)
                # read_redis_position()
                # control(sock, actuator)
                # while True:
                    # test_read_position()

            except KeyboardInterrupt:
                actuator.stop_all()

        except Exception as e:
            print(f"An error occurred during operations: {e}")
        except serial.SerialException:
            print("Failed to connect to the Arduino.")






            


            
                
