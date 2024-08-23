import numpy as np


'''
    - all values are measured in centimeters [cm] unless inches [in] are explicitly specified in variable name
    - variable notation: "x" represents a value in centimeters. "x_inches" represents a value in inches
    - wherever necessary, units will be specified for each parameter in comments using the [unit] notation (ex. [cm] for centimeters)
    - [#] represents a dimensionless numerical value
'''

# experiment parameters
YELLOW_ORBIT          = [112, 97, 81, 80, 94, 109, 125, 126]
BLACK_ORBIT           = [117, 102, 88, 89, 105, 120, 134, 133]
REF_TRAJECTORY_PERIOD = 200                                                     # total time period [sec]
SAMPLING_PERIOD       = 0.0625                                                  # camera sampling period [sec]
NUM_SAMPLES           = int(np.ceil(REF_TRAJECTORY_PERIOD / SAMPLING_PERIOD))

# platform parameters
GRID_WIDTH            = 15                                                      # grid dimensions for static dipoles [#]
NUM_COILS             = GRID_WIDTH * GRID_WIDTH
FIELD_RANGE           = 3.1                                                     # magnetic force range [cm]
COIL_SPACING          = 2.159                                                   # spacing between static dipoles: 2.159 [cm]
COERSION_THRESHOLD_IN = 0.4                                                     # a sampled position within this threshold of a coil could be coerced to coil centroid position [in]
COERSION_THRESHOLD    = COERSION_THRESHOLD_IN * 2.54                            # coersion threshold [cm]
SAMPLING_PERIOD       = 0.1                                                     # time between camera readings [sec]

# redis parameters
POSITIONS_STREAM = 'stream_positions'

# actuator parameters
ACTUATOR_PORT = "/dev/cu.usbmodem21301"

def generate_meshgrid():
    x_lower = -(GRID_WIDTH - 1) / 2
    x_upper =  (GRID_WIDTH - 1) / 2
    x_range = np.linspace(x_lower, x_upper, GRID_WIDTH) * COIL_SPACING

    y_lower = -(GRID_WIDTH - 1) / 2
    y_upper =  (GRID_WIDTH - 1) / 2
    y_range = np.linspace(y_upper, y_lower, GRID_WIDTH) * COIL_SPACING
    
    return np.meshgrid(x_range, y_range)

def generate_grid_positions():
    '''
        - generates a 2D grid, representing coil position coordinates
        - each entry represents the (x,y) coordinates of a coil on the grid
    '''

    grid_positions = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_WIDTH)]
    for i in range(GRID_WIDTH):
        for j in range(GRID_WIDTH):
            grid_positions[i][j] = np.array([X_GRID[i, j], Y_GRID[i, j]])

    return grid_positions

def generate_initial_adjacency_matrix():
    '''
        - generates initial matrix representation (called A here) of how far each coil in the platform is to its neighbors
        - A.shape: NUM_COILS x NUM_COILS (ex. a 15x15 coil platform will generate a 225x225 shaped A matrix)
        - each coil should have a total of 8 adjacent neighbors (including diagonals)
    '''

    num_coils = GRID_WIDTH * GRID_WIDTH
    A = np.zeros((num_coils, num_coils))
    # A = np.full((num_coils, num_coils), np.inf)

    for i in range(GRID_WIDTH):
        for j in range(GRID_WIDTH):
            current_idx = np.ravel_multi_index((i, j), X_GRID.shape)
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
                    neighbor_index = np.ravel_multi_index((n_i, n_j), X_GRID)
                    distance = np.linalg.norm(np.array([i, j]) - np.array([n_i, n_j]))
                    A[current_idx, neighbor_index] = distance
                    A[neighbor_index, current_idx] = distance

    return A

X_GRID, Y_GRID = generate_meshgrid()
GRID_POSITIONS = generate_grid_positions()
INITIAL_ADJACENCY_MATRIX = generate_initial_adjacency_matrix()

