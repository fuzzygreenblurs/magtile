import time
import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = 15

def create_grid():
    """Create a grid as a scatter plot."""
    fig, ax = plt.subplots()
    grid_x, grid_y = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
    ax.scatter(grid_x, grid_y, color='gray', s=10)
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)
    ax.set_xticks(np.arange(0, GRID_SIZE))
    ax.set_yticks(np.arange(0, GRID_SIZE))
    ax.set_aspect('equal')
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    black_line, = ax.plot([], [], 'o-', color='black', label='Black Trajectory')
    yellow_line, = ax.plot([], [], 'o-', color='yellow', label='Yellow Trajectory')
    # ax.legend()

    plt.ion()  # Turn on interactive mode
    plt.show()

    return fig, ax, black_line, yellow_line

def update_plot(ax, black_trajectory, yellow_trajectory, black_line, yellow_line):
    """Update the plot with new trajectories."""
    black_x, black_y = np.unravel_index(black_trajectory, (GRID_SIZE, GRID_SIZE))
    yellow_x, yellow_y = np.unravel_index(yellow_trajectory, (GRID_SIZE, GRID_SIZE))

    black_line.set_data(black_x, black_y)
    yellow_line.set_data(yellow_x, yellow_y)

    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()

def run_plot(black_trajectory, yellow_trajectory):
    """Initialize the plot and update it dynamically."""
    fig, ax, black_line, yellow_line = create_grid()

    def update_trajectories(new_black_traj, new_yellow_traj):
        update_plot(ax, new_black_traj, new_yellow_traj, black_line, yellow_line)

    return update_trajectories

# # Initialize the plotter with the first trajectory
black_trajectory = np.random.choice(GRID_SIZE**2, 10, replace=False)
yellow_trajectory = np.random.choice(GRID_SIZE**2, 10, replace=False)
update_trajectories = run_plot(black_trajectory, yellow_trajectory)

for i in range(10):
    black_trajectory = np.random.choice(GRID_SIZE**2, 10, replace=False)
    yellow_trajectory = np.random.choice(GRID_SIZE**2, 10, replace=False)
    update_trajectories(black_trajectory, yellow_trajectory)
    time.sleep(0.5)

plt.ioff()  # Turn off interactive mode when done
plt.show()