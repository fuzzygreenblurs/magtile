import numpy as np
import math
'''
    - all values are measured in centimeters [cm] unless inches [in] are explicitly specified in variable name
    - variable notation: "x" represents a value in centimeters. "x_inches" represents a value in inches
    - wherever necessary, units will be specified for each parameter in comments using the [unit] notation (ex. [cm] for centimeters)
    - [#] represents a dimensionless numerical value
'''

OPERATION_MODE = "SIMULATION" # OPTIONS: "SIMULATION" or "LIVE"
# OPERATION_MODE = "LIVE" # OPTIONS: "SIMULATION" or "LIVE"

# official experimental orbits
# experiment 1: two octagon trajectories
YELLOW_ORBIT = [40, 41, 57, 72, 86, 85, 69, 54]
BLACK_ORBIT  = [48, 63, 79, 80, 66, 51, 35, 34]

# experiment 2: two concentric octagons (check 1 coil and 2 coil range)
# YELLOW_ORBIT = [112, 97, 81, 80, 94, 109, 125, 126]
# BLACK_ORBIT  = [114, 99, 84, 68, 52, 51, 50, 49, 63, 77, 92, 107, 122, 138, 154, 155, 156, 157, 143, 129] 
# # BLACK_ORBIT  = [65, 66, 82, 98, 113, 127, 141, 140, 124, 108, 93, 79]

# # experiment 3: two intersecting infinity trajectories
# YELLOW_ORBIT = [63, 64, 65, 66, 82, 97, 112, 128, 143, 158, 174, 175, 176, 177, 163, 148, 133, 118, 103, 88, 72, 71, 70, 69, 83, 98, 113, 127, 142, 157, 171, 170, 169, 168, 152, 137, 122, 107, 92, 77]
# BLACK_ORBIT  = [94, 95, 111, 112, 128, 129, 145, 146, 131, 116, 111, 110, 114, 113, 127, 126, 140, 139, 124, 109]


# experiment parameters
# BLACK_ORBIT            = [112, 97, 81, 80, 94, 109, 125, 126]
# YELLOW_ORBIT           = [26, 27, 42, 41]
# BLACK_ORBIT            = [112, 112, 97, 97, 81, 81, 80, 80, 94, 94, 109, 109, 125, 125, 126, 126]
# YELLOW_ORBIT           = [26, 26, 27, 27, 42, 42, 41, 41]

# YELLOW_ORBIT          = [112, 112, 112, 97, 97, 97, 81, 81, 81, 80, 80, 80, 94, 94, 94, 109, 109, 109, 125, 125, 125, 126, 126, 126]
# BLACK_ORBIT           = [26, 26, 26, 27, 27, 27]
# # YELLOW_ORBIT          = [112, 97, 81, 80, 94, 109, 125, 126]
# # BLACK_ORBIT           = [117, 102, 88, 89, 105, 120, 134, 133]
REF_TRAJECTORY_PERIOD = 200                                                     # total time period [sec]
SAMPLING_PERIOD       = 0.0625                                                  # camera sampling period [sec]
NUM_SAMPLES           = int(np.ceil(REF_TRAJECTORY_PERIOD / SAMPLING_PERIOD))

# platform parameters
GRID_WIDTH            = 15                                                      # grid dimensions for static dipoles [#]
NUM_COILS             = GRID_WIDTH * GRID_WIDTH
COIL_SPACING          = 2.159                                                   # spacing between static dipoles: 2.159 [cm]
COERSION_THRESHOLD_IN = 0.4                                                     # a sampled position within this threshold of a coil could be coerced to coil centroid position [in]
SAMPLING_PERIOD       = 0.1                                                     # time between camera readings [sec]

FIELD_RANGE                = 3.5                                                # magnetic force range [cm]
OUT_OF_RANGE               = -1000000
COERSION_THRESHOLD         = COERSION_THRESHOLD_IN * 2.54                       # coersion threshold [cm]
INTERFERENCE_RANGE         = 6 * FIELD_RANGE
# DEACTIVATION_RADIUS        = 2
DEACTIVATION_RADIUS        = 2 * math.sqrt(2)                                   # [# of diagonals]
SAFE_ZONE_RADIUS           = 1
INVALIDATED_NODE_WEIGHT    = np.inf

SIMULATED_INITIAL_BLACK_POSITION = np.array([5, -5])
SIMULATED_INITIAL_YELLOW_POSITION = np.array([-6, -5])

# redis parameters
POSITIONS_STREAM = 'stream_positions'

# actuator parameters
DEFAULT_ACTUATION_DURATION = 0.5
DEFAULT_DUTY_CYCLE         = 4095
ACTUATOR_PORT = "/dev/cu.usbmodem21301"

CLUSTER_SIZE = 3
