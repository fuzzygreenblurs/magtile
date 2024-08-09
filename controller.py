import json
import numpy as np 
import networkx as nx
import math
import pdb
import time 
import serial
import socket
from actuator import Actuator

def control(queue):

    ###################################
    # Define Constants
    ###################################

    COIL_RANGE            = 3.048                    # magnetic force range: 3.048                           
    MagForce              = 1                        # magnetic force multiplier: 1                          
    x_grid_size           = 15                       # grid dimensions for static dipoles (x-direction)
    y_grid_size           = 15                       # grid dimensions for static dipoles (y-direction)
    grid_spacing          = 2.159                    # spacing between static dipoles: 2.159 (in cm)                
    
    dt                    = 0.001                    # time step
    T                     = 200                      # total time period (s)
    vt                    = np.arange(0, T, dt)      # simulation time step vector
    d                     = 0.0625                   # camera sampling period
    num_samples           = T / d                    # total number of sensor samples in the time period T 
    ticks_per_sample      = d/dt                     # time steps per sampling period

    # Initial positions and velocities of the moving dipole (disk magnet)
    x_disk, y_disk    = np.zeros(len(vt)), np.zeros(len(vt))
    vx_disk, vy_disk  = np.zeros(len(vt)), np.zeros(len(vt))

    current_position = queue.get()
    x_disk[0]        = current_position["yellow"][0]
    y_disk[0]        = current_position["yellow"][1]
    vx_disk[0]       = 0                                # Initial velocity in x-direction
    vy_disk[0]       = 0                                # Initial velocity in y-direction

    ###########################################
    # Generate Grid for Static Dipoles (Coils)
    ###########################################

    x_lower = -(x_grid_size - 1) / 2
    x_upper = (x_grid_size - 1) / 2
    x_range = np.linspace(x_lower, x_upper, x_grid_size) * grid_spacing

    y_lower = -(y_grid_size - 1) / 2
    y_upper = (y_grid_size - 1) / 2
    y_range = np.linspace(y_lower, y_upper, y_grid_size) * grid_spacing

    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = np.zeros(len(x_range))

    # generate a 2D grid, representing coil position coordinates
    xy_grid_cells = [[None for _ in range(y_grid_size)] for _ in range(x_grid_size)]
    for i in range(x_grid_size):
        for j in range(y_grid_size):
            xy_grid_cells[i][j] = np.array([x_grid[i, j], y_grid[i, j]])

    ###################################
    # Create Adjacency Matrix A
    ###################################

    # create a 225x225 grid to store the distance from each vertex to its neighbors
    num_coils = x_grid_size * y_grid_size
    A = np.zeros((num_coils, num_coils))

    for i in range(x_grid_size):
        for j in range(y_grid_size):
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
                if 0 <= n_i < x_grid_size and 0 <= n_j < y_grid_size:
                    neighbor_index = np.ravel_multi_index((n_i, n_j), x_grid.shape)
                    distance = np.linalg.norm(np.array([i, j]) - np.array([n_i, n_j]))
                    A[current_idx, neighbor_index] = distance
                    A[neighbor_index, current_idx] = distance

    ###########################################
    # Initialize Reference Signals
    ###########################################

    # kernel: just describes a single (non-repeating) unit
    ref_trajectory_kernel_idx = [19, 20, 35, 49, 48, 33]
    ref_trajectory_kernel_subscript = np.unravel_index(ref_trajectory_kernel_idx, x_grid.shape)
    expected_input_kernel_idx = ref_trajectory_kernel_idx

    # use np.tile() to repeat the kernel arrays enough times to cover the total number of control steps
    ref_trajectory = np.tile(ref_trajectory_kernel_subscript, 1, int(np.ceil(T/d)))
    expected_input_idx = np.tile(expected_input_kernel_idx, int(np.ceil(T/d)))

    # slice the repetitions to match the number of samples exactly (including the initial step)
    ref_trajectory = ref_trajectory[:, int((T/d) + 1)]

    # this is just the target loop by idx: 
    # [19, 20, 35, 49, 48, 33 | 19, 20, 35, 49, 48, 33 | 19, 20, 35, 49, 48, 33 | ... ]
    expected_input_idx = expected_input_idx[:, int((T/d) + 1)]

    ###########################################
    # Control Loop
    ###########################################

    # define control and error signal vectors
    actual_input = np.zeros(len(vt))
    error        = np.zeros(len(vt))

    for s in range(int(T/d)):
        # observed position at the start of the current sample interval
        prior_raw = queue.get()["yellow"]
        prior_position = np.array([prior_raw[0], prior_raw[1]])
        
        ref_posterior_position = ref_trajectory[:, s + 1]
        sample_error = np.linalg.norm(prior_position - ref_posterior_position)

        if sample_error <= COIL_RANGE:
            # set the input to the expected control signal for the target coil
            target_coil_idx = expected_input_idx[s+1]
            target_coil = np.unravel_index(target_coil_idx, x_grid.shape)
            actuator.actuate_single(target_coil[0], target_coil[1])
        else:
            min_dist = math.inf
            min_idx = 0

            # find the current closest coil in the grid
            for i in num_coils:
                xy_cell = xy_grid_cells[i]
                error_cell = np.linalg.norm(prior_position - xy_cell)
                if (error_cell < min_dist):
                    min_dist = error_cell
                    min_idx = i

            # compute shortest path to the reference trajectory: Djikstra
            graph = nx.from_numpy_array(A)
            sample_input_sequence = nx.dijkstra_path(graph, min_idx, target_coil_idx)
            sample_input_sequence = sample_input_sequence[::-1]

            sample_input_1, sample_input_2 = np.zeros(num_coils, 1), np.zeros(num_coils, 1)
            
            # if the disk is within the radius of the second coil in the expected shortest path, skip directly to it
            if np.linalg.norm(prior_position - xy_grid_cells[sample_input_sequence[1]]) <= COIL_RANGE:
                target_coil_idx = sample_input_sequence[1]
                target_coil = np.unravel_index(target_coil_idx, x_grid.shape)
                actuator.actuate_single(target_coil[0], target_coil[1])
            else:
                target_coil_idx = sample_input_sequence[0]
                target_coil = np.unravel_index(target_coil_idx, x_grid.shape)
                actuator.actuate_single(target_coil[0], target_coil[1])

def control_test(sock):
    while True:
        data, _ = sock.recvfrom(1024)  # Buffer size is 1024 bytes
        payload = json.loads(data.decode())
        print("yellow x:", payload["yellow"][0], "yellow y:", payload["yellow"][1])

if __name__ == '__main__':
    server_address = ('localhost', 65432)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(server_address)
        control_test(sock)



    # port = "/dev/cu.usbmodem11301"  # Update with the correct port for your setup
    # try:
    #     with Actuator(port) as actuator:
    #         try:
    #             # Read dimensions
    #             width = actuator.read_width()
    #             height = actuator.read_height()
    #             print(f"Width: {width}, Height: {height}")

    #             addresses = actuator.scan_addresses()
    #             print(f"Addresses: {addresses}")

    #             # Store configuration
    #             actuator.store_config()

    #             try:
    #                 control(q, actuator)
    #             except KeyboardInterrupt:
    #                 actuator.stop_all()

    #         except Exception as e:
    #             print(f"An error occurred during operations: {e}")
    # except serial.SerialException:
    #     print("Failed to connect to the Arduino.")






            


            
                
