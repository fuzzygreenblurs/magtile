import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import pdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
MM_TO_IN = 0.0394
CASE_HEIGHT = 12.50 * MM_TO_IN
COIL_HEIGHT = 12.03 * MM_TO_IN
SHEET_THICKNESS = 3.485 * MM_TO_IN
DISC_HEIGHT = 1.75 * MM_TO_IN
R1 = 9.455 * MM_TO_IN          # coil radius
R2 = 2.965 * MM_TO_IN          # disc radius
z1 = 0
z2 = (COIL_HEIGHT / 2) + (CASE_HEIGHT - COIL_HEIGHT) + (2 * SHEET_THICKNESS) + (DISC_HEIGHT / 2)
h = z2 - z1
frame_rate = 83
time_step = 1 / frame_rate

# Load csvs to df
HOME = '/Users/akhilsankar/workspace/swarms/tracking'
TARGET = 'regression_tracked_actions/tracked_segmented'

def generate_finite_differences(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['x2', 'y2']
    global POSITION_A, POSITION_B
    POSITION_A = [round(df.head(3)['x2'].mean(), 3), round(df.head(3)['y2'].mean(), 3)]
    POSITION_B = [round(df.tail(3)['x2'].mean(), 3), round(df.tail(3)['y2'].mean(), 3)]

    df['vx2'] = np.round(df['x2'].diff() / time_step, 2)
    df['vy2'] = np.round(df['y2'].diff() / time_step, 2)
    df.loc[0, 'vx2'] = 0
    df.loc[0, 'vy2'] = 0

    return df

def state_space_model(params, x2_prior, y2_prior, vx2_prior, vy2_prior):
    alpha, beta = params
    r = np.sqrt(((x2_prior - POSITION_B[0])**2) + ((y2_prior - POSITION_B[1])**2) + (h**2))

    ax2 = -beta * vx2_prior + (alpha / (r**5)) * (1 - 5 * (h**2) / (r**2)) * (x2_prior - POSITION_B[0])
    vx2_post = vx2_prior + (ax2 * time_step)
    x2_post = x2_prior + (vx2_post * time_step)

    ay2 = -beta * vy2_prior + (alpha / (r**5)) * (1 - 5 * (h**2) / (r**2)) * (y2_prior - POSITION_B[1])
    vy2_post = vy2_prior + (ay2 * time_step)
    y2_post = y2_prior + (vy2_post * time_step)

    return x2_post, y2_post

def objective_function(inputs, alpha, beta):
    params = [alpha, beta]
    x2, y2, vx2, vy2 = inputs
    x2_predicted, y2_predicted = state_space_model(params, x2[:-1], y2[:-1], vx2[:-1], vy2[:-1])
    return np.concatenate([x2_predicted, y2_predicted])

def perform_regression(file_path, initial_guesses=[0.72, 1]):
    df = generate_finite_differences(file_path)
    
    inputs = np.array([df['x2'].values, df['y2'].values, df['vx2'].values, df['vy2'].values])
    observed_posterior_positions = df.loc[1:, ['x2', 'y2']].values.flatten()

    try:
        optimized_params, covariance = curve_fit(
            lambda inputs, alpha, beta: objective_function(inputs, alpha, beta),
            inputs,
            observed_posterior_positions,
            p0=initial_guesses
        )
        return optimized_params

    except Exception as e:
        print("Error during curve fitting:", e)
        return None

# Directory paths
DIR_PATH = f"{HOME}/{TARGET}"

# List to store results
results = []

# Process each file and collect results
for vertex in os.listdir(DIR_PATH):
    vertex_path = os.path.join(DIR_PATH, vertex)
    for direction in os.listdir(vertex_path):
        if direction not in [f"center_to_{vertex}", f"{vertex}_to_center"]:
            continue

        action_path = os.path.join(vertex_path, direction, "trimmed")

        for _, _, segments in os.walk(action_path):
            for segment_id in segments:
                segment_path = os.path.join(action_path, segment_id)
                params = perform_regression(segment_path)
                if params is not None:
                    results.append(params.tolist())
                print(
                    "vertex: ", vertex,
                    "action: ", direction,
                    "params: ", params
                )
        print(f"\n")

# Create a DataFrame from results and save to CSV
results_df = pd.DataFrame(results, columns=['alpha', 'beta'])
results_df.to_csv(f"{HOME}/bare_optimized_params.csv", index=False)

print("Optimized parameters saved to CSV.")


################## BOX PLOT #############################################

plt.figure(figsize=(10, 6))
sns.boxplot(y=results_df['alpha'])
plt.title('Box Plot of Alpha')
plt.ylabel('Alpha')
plt.ylim(results_df['alpha'].min() - 0.1, results_df['alpha'].max() + 0.1)
# Annotate min and max values
min_val = round(results_df['alpha'].min(), 2)
max_val = round(results_df['alpha'].max(), 2)
plt.annotate(f'Min: {min_val}', xy=(0, min_val), xytext=(0.1, min_val - 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate(f'Max: {max_val}', xy=(0, max_val), xytext=(0.1, max_val + 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
