import pdb
import time
import numpy as np
import matplotlib.pyplot as plt
from sim_platform import Platform
from shared_data import shared_data
import threading

GRID_SIZE = 15

def create_grid():
    """Create a grid as a scatter plot."""
    fig, ax = plt.subplots()
    grid_y, grid_x = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))

    ax.scatter(grid_x, grid_y, color='gray', s=10)
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(GRID_SIZE, -1)  # Invert the y-axis to place the origin at the top left
    ax.set_xticks(np.arange(0, GRID_SIZE))
    ax.set_yticks(np.arange(0, GRID_SIZE))
    ax.set_aspect('equal')
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    black_ref_line, = ax.plot([], [], 'o-', color='purple', label='black ref trajectory')
    yellow_ref_line, = ax.plot([], [], 'o-', color='red', label='yellow ref trajectory')
    black_line = ax.plot([], [], 'o-', color='black', label='black Trajectory')
    yellow_line = ax.plot([], [], 'o-', color='orange', label='yellow Trajectory')
    ax.legend()

    plt.ion()  # Turn on interactive mode
    plt.show()

    return fig, ax, black_line, yellow_line, black_ref_line, yellow_ref_line

def update_plot(ax, black_trajectory, yellow_trajectory, black_ref_trajectory, yellow_ref_trajectory, black_line, yellow_line, black_ref_line, yellow_ref_line, platform):
    """Update the plot with new trajectories."""
    
    # Clear the entire plot and re-plot the grid
    ax.cla()
    # deactivated_positions = platform.deactivated_positions
    grid_y, grid_x = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
    flat_grid_x = grid_x.flatten()
    flat_grid_y = grid_y.flatten()

    # Print debug information
    # print("Deactivated Positions (Flattened Indices):", deactivated_positions)
    # print("flat_grid_x:", flat_grid_x)
    # print("flat_grid_y:", flat_grid_y)

    # # Determine the valid and invalid positions
    valid_mask = np.ones(flat_grid_x.shape, dtype=bool)
    # valid_mask[deactivated_positions] = False

    # # Plot valid positions in gray
    ax.scatter(flat_grid_x[valid_mask], flat_grid_y[valid_mask], color='gray', s=10, label='Active Positions')
    
    # # Plot invalid (deactivated) positions in red
    # ax.scatter(flat_grid_x[~valid_mask], flat_grid_y[~valid_mask], color='red', s=50, marker='x', label='Deactivated Positions')

    # ax.scatter(grid_x, grid_y, color='gray', s=10)
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(GRID_SIZE, -1)  # Invert the y-axis to place the origin at the top left
    ax.set_xticks(np.arange(0, GRID_SIZE))
    ax.set_yticks(np.arange(0, GRID_SIZE))
    ax.set_aspect('equal')

    # Convert grid indices to coordinates for plotting
    black_ref_y, black_ref_x = np.unravel_index(black_ref_trajectory, (GRID_SIZE, GRID_SIZE))
    yellow_ref_y, yellow_ref_x = np.unravel_index(yellow_ref_trajectory, (GRID_SIZE, GRID_SIZE))
    black_y, black_x = np.unravel_index(black_trajectory, (GRID_SIZE, GRID_SIZE))
    yellow_y, yellow_x = np.unravel_index(yellow_trajectory, (GRID_SIZE, GRID_SIZE))

    # Plot the reference trajectories
    black_ref_line = ax.plot(black_ref_x, black_ref_y, 'o-', color='purple', label='black ref trajectory')
    yellow_ref_line = ax.plot(yellow_ref_x, yellow_ref_y, 'o-', color='red', label='yellow ref trajectory')

    # Plot the first point of the black trajectory in a different color
    ax.plot(black_x[0], black_y[0], 'o', color='black', markersize=15)
    ax.plot(yellow_x[0], yellow_y[0], 'o', color='orange', markersize=15)

    # Plot the remaining points of the black trajectory
    black_line = ax.plot(black_x, black_y, 'o-', color='black', label='black trajectory')

    # Plot the yellow trajectory
    yellow_line = ax.plot(yellow_x, yellow_y, 'o-', color='orange', label='yellow trajectory')
    
    ax.legend()

    # Redraw the canvas with updated data
    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()

def run_plot(black_trajectory, yellow_trajectory, black_ref_trajectory,  yellow_ref_trajectory, platform):
    """Initialize the plot and update it dynamically."""
    fig, ax, black_line, yellow_line, black_ref_line, yellow_ref_line = create_grid()

    def update_trajectories(new_black_traj, new_yellow_traj, new_black_ref_trajectory, new_yellow_ref_trajectory, platform):
        update_plot(ax, new_black_traj, new_yellow_traj, new_black_ref_trajectory, new_yellow_ref_trajectory, black_line, yellow_line, black_ref_line, yellow_ref_line, platform)

    return update_trajectories

def run_control_loop(platform):
    platform.control()

if __name__ == "__main__":
    platform = Platform()
    platform.deactivated_neighbors = []

    black_ref_trajectory = platform.black_agent.ref_trajectory
    black_input_trajectory = platform.black_agent.shortest_path
    yellow_ref_trajectory = platform.yellow_agent.ref_trajectory
    yellow_input_trajectory = platform.yellow_agent.shortest_path
    
    # black_ref_trajectory = platform.black_agent.ref_trajectory
    # black_input_trajectory = platform.black_agent.input_trajectory[0:2]
    # yellow_ref_trajectory = platform.yellow_agent.ref_trajectory
    # yellow_input_trajectory = platform.yellow_agent.input_trajectory[0:2]

    update_trajectories = run_plot(black_input_trajectory, yellow_input_trajectory, black_ref_trajectory, yellow_ref_trajectory, platform)

    i = 0
    while True:
        platform.deactivated_neighbors = []
        platform.current_control_iteration = i           
        print(f"\n--- control loop: {i} ----")
        
        platform.reset_agent_flags()
        platform.update_all_agent_positions()
        platform.plan_for_interference()
        platform.advance_agents()

        black_input_trajectory = platform.black_agent.shortest_path
        yellow_input_trajectory = platform.yellow_agent.shortest_path
        update_trajectories(black_input_trajectory, yellow_input_trajectory, black_ref_trajectory, yellow_ref_trajectory, platform)
        plt.pause(0.01)
        
        i += 1

    plt.ioff()  # Turn off interactive mode when done
    plt.show()