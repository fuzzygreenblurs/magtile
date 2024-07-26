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
position_A = np.array([2.48, -0.67])
position_B = np.array([1.58, -1.58])
frame_rate = 83
time_step = 1 / frame_rate

## load csvs to df
df = pd.read_csv('ur/ur_to_center/segment_1_trimmed.csv')
positionA = [df.head(3)['h_pos'].mean(), df.head(3)['v_pos'].mean()]
positionB = [df.tail(3)['h_pos'].mean(), df.tail(3)['v_pos'].mean()]

new_column_names = ['x2', 'y2']
df.columns = new_column_names
df.ignore_index = True

## compute velocities
SAME_POINT_THRESHOLD = 0.02 
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

## define state space model and objective functions 
def state_space_model_x(params, x0, vx0, t):
    alpha, beta1 = params
    r = np.sqrt((x0 - position_B[0])**2 + h**2)
    ax2_pred = -beta1 * vx0 + (alpha / r**5) * (1 - 5 * (h**2) / r**2) * (x0 - position_B[0])
    vx1 = vx0 + ax2_pred * t
    x1 = x0 + vx1 * t
    
    return x1

def objective_function_x(params, x2, vx2, t, x_obs):
    x_pred = state_space_model_x(params, x2, vx2, t)
    residuals = x_obs - x_pred
    return residuals

def wrapped_objective_function_x(inputs, alpha, beta1):
    params = [alpha, beta1]
    x2, vx2 = inputs
    t = np.arange(len(x2)) * time_step
    return objective_function_x(params, x2, vx2, t, df['x2'])

## perform regression

initial_guesses = [0.01, 0.01]
inputs = np.array([df['x2'], df['vx2']])

try:
    optimized_params, covariance = curve_fit(
        wrapped_objective_function_x,
        inputs,
        df['x2'],
        p0=initial_guesses
    )
    print("Optimized parameters:", optimized_params)
except Exception as e:
    print("Error during curve fitting:", e)

## perform simulation
time_step = 0.0001
alpha_opt, beta1_opt = optimized_params  # Replace with optimized parameters
print(position_A, position_B)

# Define the state-space model for the x position
def state_space_model_x(params, x2, vx2):
    alpha, beta1 = params
    r = np.sqrt((x2 - position_B[0])**2 + h**2)
    ax2_pred = -beta1 * vx2 + (alpha / r**5) * (1 - 5 * (h**2) / r**2) * (x2 - position_B[0])
    return ax2_pred

# Simulation parameters  # Time step (adjust based on your data)
num_steps = 10000  # Number of steps in the simulation

# Initial conditions
x2_pred = position_A[0]  # Initial position
vx2_pred = 0.0  # Initial velocity

# Simulation loop using Euler method
for i in range(num_steps):
    ax2_pred  = state_space_model_x([alpha_opt, beta1_opt], x2_pred, vx2_pred)
    vx2_pred += ax2_pred * time_step
    x2_pred  += vx2_pred * time_step
    print(x2_pred)