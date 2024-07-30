## define imports and constants

import cv2
import numpy as np
import pdb
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

MM_TO_IN            = 0.0394
CASE_HEIGHT         = 12.50 * MM_TO_IN
COIL_HEIGHT         = 12.03 * MM_TO_IN
SHEET_THICKNESS     = 3.485 * MM_TO_IN
DISC_HEIGHT         = 1.75  * MM_TO_IN
R1                  = 9.455 * MM_TO_IN          # coil radius
R2                  = 2.965 * MM_TO_IN          # disc radius
z1 = 0
z2 = (COIL_HEIGHT / 2) + (CASE_HEIGHT - COIL_HEIGHT) + (2 * SHEET_THICKNESS) + (DISC_HEIGHT / 2)
h = z2 - z1
frame_rate = 83
time_step = 1 / frame_rate

## load csvs to df
TARGET = 'regression_tracked_actions/tracked_segmented'
VERTEX = 'ur'
# ACTION = f"center_to_{VERTEX}"
ACTION = f"{VERTEX}_to_center"
SEGMENT_ID = "segment_1"

df = pd.read_csv(f"{TARGET}/{VERTEX}/{ACTION}/trimmed/{SEGMENT_ID}.csv")
df.columns = ['x2', 'y2']
position_A = [round(df.head(3)['x2'].mean(), 3), round(df.head(3)['y2'].mean(), 3)]
position_B = [round(df.tail(3)['x2'].mean(), 3), round(df.tail(3)['y2'].mean(), 3)]
df.ignore_index = True

## compute velocities
# SAME_POINT_THRESHOLD = 0.02 
frame_rate = 83
time_step = 1 / frame_rate

# Calculate the velocity components starting from the second row
df['vx2'] = np.round(df['x2'].diff() / time_step, 2)
df['vy2'] = np.round(df['y2'].diff() / time_step, 2)

# Fill the first row of velocity with zeros (since diff will produce NaN for the first element)
df.loc[0, 'vx2'] = 0
df.loc[0, 'vy2'] = 0

## store df as csv
output_file_path = '/Users/akhilsankar/workspace/swarms/tracking/combined_static_trial_dfs/ur/segment_1_trimmed.csv'
df.to_csv(output_file_path, index=False)

############################################################### REGRESSION ############################################################
############################################################### REGRESSION ############################################################
############################################################### REGRESSION ############################################################
############################################################### REGRESSION ############################################################

## define state space model and objective functions 
# Define the state-space model for x-position
def state_space_model(params, x2_prior, y2_prior, vx2_prior, vy2_prior):
    alpha, beta = params
    r = np.sqrt(((x2_prior - position_B[0])**2) + ((y2_prior - position_B[1])**2) + (h**2))

    ax2      = -beta * vx2_prior + (alpha / (r**5)) * (1 - 5 * (h**2) / (r**2)) * (x2_prior - position_B[0])
    vx2_post = vx2_prior + (ax2 * time_step)
    x2_post  = x2_prior + (vx2_post  * time_step)

    ay2      = -beta * vy2_prior + (alpha / (r**5)) * (1 - 5 * (h**2) / (r**2)) * (y2_prior - position_B[1])
    vy2_post = vy2_prior + (ay2 * time_step)
    y2_post  = y2_prior + (vy2_post  * time_step)

    return x2_post, y2_post

# Define the combined objective function
def objective_function(inputs, alpha, beta):
    params = [alpha, beta]
    x2, y2, vx2, vy2 = inputs
    x2_predicted, y2_predicted = state_space_model(params, x2[0:-1], y2[0:-1], vx2[0:-1], vy2[0:-1])

    return np.concatenate([x2_predicted, y2_predicted])

# inputs = np.array([df['x2'], df['y2'], df['vx2'], df['vy2']])
# initial_guesses = [0.72, 1]

# observed_posterior_positions = df.loc[1:, ['x2', 'y2']]
# observed_posterior_positions = df.loc[1:, ['x2', 'y2']].values.flatten()

# try:
#     optimized_params, covariance = curve_fit(
#         objective_function,
#         inputs,
#         observed_posterior_positions,
#         p0=initial_guesses
#     )

#     print(
#         "vertex: ", VERTEX, "\n",
#         "action: ", ACTION, "\n",
#         "segment_id: ", SEGMENT_ID, "\n",
#         "initial parameters:", initial_guesses, "\n",
#         "optimized parameters:", optimized_params
#     )

# except Exception as e:
#     print("Error during curve fitting:", e)

def perform_regression(df, initial_guesses=[0.72, 1]):
    inputs = np.array([df['x2'], df['y2'], df['vx2'], df['vy2']])
    initial_guesses = initial_guesses
    observed_posterior_positions = df.loc[1:, ['x2', 'y2']].values.flatten()

    try:
        optimized_params, covariance = curve_fit(
            objective_function,
            inputs,
            observed_posterior_positions,
            p0=initial_guesses
        )

        print(
            "vertex: ", VERTEX, "\n",
            "action: ", ACTION, "\n",
            "segment_id: ", SEGMENT_ID, "\n",
            "initial parameters:", initial_guesses, "\n",
            "optimized parameters:", optimized_params
        )

    except Exception as e:
        print("Error during curve fitting:", e)

############################################################### SIMULATION ############################################################
############################################################### SIMULATION ############################################################
############################################################### SIMULATION ############################################################
############################################################### SIMULATION ############################################################

# time_step = 0.001
# alpha_opt, beta1_opt = optimized_params

# def sim_state_space_model(params, x2_prior, y2_prior, vx2_prior, vy2_prior):
#     alpha, beta = params
#     r = np.sqrt(((x2_prior - position_B[0])**2) + ((y2_prior - position_B[1])**2) + (h**2))
#     ax2 = -beta * vx2_prior + (alpha / (r**5)) * (1 - 5 * (h**2) / (r**2)) * (x2_prior - position_B[0])
#     ay2 = -beta * vy2_prior + (alpha / (r**5)) * (1 - 5 * (h**2) / (r**2)) * (y2_prior - position_B[1])
#     return ax2, ay2

# # Simulation parameters  # Time step (adjust based on your data)
# num_steps = 1500  # Number of steps in the simulation

# # Initial conditions
# x2_pred, y2_pred   = position_A[0], position_A[1] 
# vx2_pred, vy2_pred = 0.0, 0.0
# print("start position: ", x2_pred, vx2_pred)
# print("target end position: ", position_B[0], position_B[1])

# # Simulation loop using Euler method
# for i in range(num_steps):
#     ax2_pred, ay2_pred  = sim_state_space_model(
#         [alpha_opt, beta1_opt], 
#         x2_pred, vx2_pred, 
#         y2_pred, vy2_pred
#     )

#     vx2_pred += ax2_pred * time_step
#     x2_pred  += vx2_pred * time_step
#     vy2_pred += ax2_pred * time_step
#     y2_pred  += vx2_pred * time_step

# print(
#     "optimized params", optimized_params, "\n",
#     "start positition: ", position_A[0], "\n",
#     "target end position: ", position_B[0], "\n",
#     "actual end position: ", x2_pred, y2_pred
#     )