import numpy as np 
import pdb

###################################
# Define Constants
###################################

alpha                 = 360                      # magnetic force coefficient: 360
beta                  = 9                        # damping coefficient: 9
MagRng                = 3.048                    # magnetic force range: 3.048                           
MagForce              = 1                        # magnetic force multiplier: 1                          
x_grid_size           = 15                       # grid dimensions for static dipoles (x-direction)
y_grid_size           = 15                       # grid dimensions for static dipoles (y-direction)
grid_spacing          = 2.159                    # spacing between static dipoles: 2.159 (in cm)                

dt                    = 0.001                    # time step
T                     = 200                      # total time period
vt                    = np.arange(0, T, dt)      # simulation time step vector
d                     = 10                       # camera sampling period
steps                 = d/dt                     # time steps per sampling period

# Initial positions and velocities of the moving dipole (disk magnet)
x_disk, y_disk    = np.zeros(len(vt)), np.zeros(len(vt))
vx_disk, vy_disk  = np.zeros(len(vt)), np.zeros(len(vt))

z_disk       = 1.435       # Height of the plane above the origin
x_disk[0]    = None        # TODO: set the initial position using sensor reading
y_disk[0]    = None        # TODO: set the initial position using sensor reading
vx_disk[0]   = 0           # Initial velocity in x-direction
vy_disk[0]   = 0           # Initial velocity in y-direction

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

num_coils = x_grid_size * y_grid_size
A = np.zeros((x_grid_size, y_grid_size))

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
            print(n_j)
            if 0 <= n_i < x_grid_size and 0 <= n_j < y_grid_size:
                neighbor_index = np.ravel_multi_index((n_i, n_j), x_grid.shape)
                distance = np.linalg.norm(np.array([i, j]) - np.array([n_i, n_j]))
                A[current_idx, neighbor_index] = distance
                A[neighbor_index, current_idx] = distance

pdb.set_trace()