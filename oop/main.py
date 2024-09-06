import pdb
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import redis
from agent import Agent
from magtile_platform import Platform
from actuator import Actuator
from constants import *

def create_grid():
    """Create a grid as a scatter plot."""
    fig, ax = plt.subplots()
    grid_y, grid_x = np.meshgrid(np.arange(GRID_WIDTH), np.arange(GRID_WIDTH))

    ax.scatter(grid_x, grid_y, color='gray', s=10)
    ax.set_xlim(-1, GRID_WIDTH)
    ax.set_ylim(GRID_WIDTH, -1)  # Invert the y-axis to place the origin at the top left
    ax.set_xticks(np.arange(0, GRID_WIDTH))
    ax.set_yticks(np.arange(0, GRID_WIDTH))
    ax.set_aspect('equal')
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    black_ref_line, = ax.plot([], [], 'o-', color='purple', label='black ref trajectory')
    yellow_ref_line, = ax.plot([], [], 'o-', color='red', label='yellow ref trajectory')
    black_line = ax.plot([], [], 'o-', color='black', label='black Trajectory')
    yellow_line = ax.plot([], [], 'o-', color='orange', label='yellow Trajectory')
    # ax.legend()

    plt.ion()  # Turn on interactive mode
    plt.show()

    return fig, ax, black_line, yellow_line, black_ref_line, yellow_ref_line

def update_plot(ax, black_trajectory, yellow_trajectory, black_ref_trajectory, yellow_ref_trajectory, black_line, yellow_line, black_ref_line, yellow_ref_line, platform):
    """Update the plot with new trajectories."""
    
    # Clear the entire plot and re-plot the grid
    ax.cla()
    grid_y, grid_x = np.meshgrid(np.arange(GRID_WIDTH), np.arange(GRID_WIDTH))
    flat_grid_x = grid_x.flatten()
    flat_grid_y = grid_y.flatten()

    deactivated_indices = [np.ravel_multi_index((y, x), (GRID_WIDTH, GRID_WIDTH)) for x, y in platform.deactivated_positions]
    deactivated_x = flat_grid_x[deactivated_indices]
    deactivated_y = flat_grid_y[deactivated_indices]
    ax.scatter(deactivated_x, deactivated_y, color='red', s=50, marker='x', label='Deactivated Positions')

    # # Determine the valid and invalid positions
    valid_mask = np.ones(flat_grid_x.shape, dtype=bool)
    ax.scatter(flat_grid_x[valid_mask], flat_grid_y[valid_mask], color='gray', s=10, label='Active Positions')

    ax.set_xlim(-1, GRID_WIDTH)
    ax.set_ylim(GRID_WIDTH, -1)  # Invert the y-axis to place the origin at the top left
    ax.set_xticks(np.arange(0, GRID_WIDTH))
    ax.set_yticks(np.arange(0, GRID_WIDTH))
    ax.set_aspect('equal')

    # Convert grid indices to coordinates for plotting
    black_ref_y, black_ref_x = np.unravel_index(black_ref_trajectory, (GRID_WIDTH, GRID_WIDTH))
    yellow_ref_y, yellow_ref_x = np.unravel_index(yellow_ref_trajectory, (GRID_WIDTH, GRID_WIDTH))
    black_y, black_x = np.unravel_index(black_trajectory, (GRID_WIDTH, GRID_WIDTH))
    yellow_y, yellow_x = np.unravel_index(yellow_trajectory, (GRID_WIDTH, GRID_WIDTH))

    # Plot the reference trajectories
    black_ref_line = ax.plot(black_ref_x, black_ref_y, 'o-', color='purple', label='black ref trajectory')
    yellow_ref_line = ax.plot(yellow_ref_x, yellow_ref_y, 'o-', color='red', label='yellow ref trajectory')

    # Plot the first point of the black trajectory in a different color
    if platform.black_agent.shortest_path:
        ax.plot(black_x[0], black_y[0], 'o', color='black', markersize=15)
    else:
        ax.plot(black_x[i], black_y[i], 'o', color='black', markersize=15)    
    
    if platform.yellow_agent.shortest_path:
        ax.plot(yellow_x[0], yellow_y[0], 'o', color='orange', markersize=15)
    else:
        ax.plot(yellow_x[i], yellow_y[i], 'o', color='orange', markersize=15)    

    # ax.plot(black_x[i], black_y[i], 'o', color='black', markersize=15)
    # ax.plot(yellow_x[i], yellow_y[i], 'o', color='orange', markersize=15)

    # Plot the remaining points of the black trajectory
    black_line = ax.plot(black_x, black_y, 'o-', color='black', label='black trajectory')

    # Plot the yellow trajectory
    yellow_line = ax.plot(yellow_x, yellow_y, 'o-', color='orange', label='yellow trajectory')
    
    # ax.legend()

    # Redraw the canvas with updated data
    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()

def run_plot(black_trajectory, yellow_trajectory, black_ref_trajectory,  yellow_ref_trajectory, platform):
    """Initialize the plot and update it dynamically."""
    fig, ax, black_line, yellow_line, black_ref_line, yellow_ref_line = create_grid()

    def update_trajectories(new_black_traj, new_yellow_traj, new_black_ref_trajectory, new_yellow_ref_trajectory, platform):
        update_plot(ax, new_black_traj, new_yellow_traj, new_black_ref_trajectory, new_yellow_ref_trajectory, black_line, yellow_line, black_ref_line, yellow_ref_line, platform)

    return update_trajectories
 
if __name__ == '__main__':
    with Actuator("/dev/cu.usbmodem21301") as actuator:
        with redis.Redis(host='localhost', port=6379, db=0) as ipc_client:
            Agent.set_actuator(actuator)
            platform = Platform(ipc_client)
            # print("initial state: ")
            # for a in platform.agents:
            #         print(f"{a.color}: {a.position}")

            black_ref_trajectory = platform.black_agent.ref_trajectory
            black_input_trajectory = platform.black_agent.shortest_path
            yellow_ref_trajectory = platform.yellow_agent.ref_trajectory
            yellow_input_trajectory = platform.yellow_agent.shortest_path

            update_trajectories = run_plot(black_input_trajectory, yellow_input_trajectory, black_ref_trajectory, yellow_ref_trajectory, platform)

            global i 
            i = 0

            while True:
                platform.current_control_iteration = i           
                print(f"\n--- control loop: {i} ----")  

                platform.reset_interference_parameters()
                platform.update_agent_positions()

                # if np.linalg.norm(platform.yellow_agent.position - platform.black_agent.position) <= 2 * COIL_SPACING:
                #     pdb.set_trace()

                platform.plan_for_interference()

                print("black sp: ", platform.black_agent.shortest_path)
                print("yellow sp: ", platform.yellow_agent.shortest_path)

                asyncio.run(platform.advance_agents())

                black_input_trajectory = platform.black_agent.shortest_path
                yellow_input_trajectory = platform.yellow_agent.shortest_path

                if platform.black_agent.shortest_path:
                    black_input_trajectory = platform.black_agent.shortest_path
                else:
                    black_input_trajectory = platform.black_agent.input_trajectory
                
                if platform.yellow_agent.shortest_path:
                    yellow_input_trajectory = platform.yellow_agent.shortest_path
                else:
                    yellow_input_trajectory = platform.yellow_agent.input_trajectory

                update_trajectories(black_input_trajectory, yellow_input_trajectory, black_ref_trajectory, yellow_ref_trajectory, platform)
                # plt.pause(1)

                if i == 9:
                    pdb.set_trace()

                i += 1

            plt.ioff()  # Turn off interactive mode when done
            plt.show()
        