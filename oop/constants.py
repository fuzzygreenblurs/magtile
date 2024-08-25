import numpy as np

'''
    - all values are measured in centimeters [cm] unless inches [in] are explicitly specified in variable name
    - variable notation: "x" represents a value in centimeters. "x_inches" represents a value in inches
    - wherever necessary, units will be specified for each parameter in comments using the [unit] notation (ex. [cm] for centimeters)
    - [#] represents a dimensionless numerical value
'''

# experiment parameters
# YELLOW_ORBIT          = [112, 112, 112, 97, 97, 97, 81, 81, 81, 80, 80, 80, 94, 94, 94, 109, 109, 109, 125, 125, 125, 126, 126, 126]
YELLOW_ORBIT          = [112, 112, 97, 97, 81, 81, 80, 80, 94, 94, 109, 109, 125, 125, 126, 126]
BLACK_ORBIT           = [26, 27]
# YELLOW_ORBIT          = [112, 97, 81, 80, 94, 109, 125, 126]
# BLACK_ORBIT           = [117, 102, 88, 89, 105, 120, 134, 133]
REF_TRAJECTORY_PERIOD = 200                                                     # total time period [sec]
SAMPLING_PERIOD       = 0.0625                                                  # camera sampling period [sec]
NUM_SAMPLES           = int(np.ceil(REF_TRAJECTORY_PERIOD / SAMPLING_PERIOD))

# platform parameters
GRID_WIDTH            = 15                                                      # grid dimensions for static dipoles [#]
NUM_COILS             = GRID_WIDTH * GRID_WIDTH
COIL_SPACING          = 2.159                                                   # spacing between static dipoles: 2.159 [cm]
COERSION_THRESHOLD_IN = 0.4                                                     # a sampled position within this threshold of a coil could be coerced to coil centroid position [in]
SAMPLING_PERIOD       = 0.1                                                     # time between camera readings [sec]

FIELD_RANGE                = 3.5                                                       # magnetic force range [cm]
OUT_OF_RANGE               = -1000000
COERSION_THRESHOLD         = COERSION_THRESHOLD_IN * 2.54                            # coersion threshold [cm]
INTERFERENCE_RANGE         = 3 * FIELD_RANGE
DEFAULT_ACTUATION_DURATION = 0.4
DEFAULT_DUTY_CYCLE         = 4095

# redis parameters
POSITIONS_STREAM = 'stream_positions'

# actuator parameters
ACTUATOR_PORT = "/dev/cu.usbmodem21301"