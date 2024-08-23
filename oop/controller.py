import redis
import time
import numpy as np
from agent_color import AgentColor
from agent import Agent
from platform import Platform
from actuator import Actuator

if __name__ == '__main__':
    '''
    note: the term ipc_client is used to descibe a generic interprocess communication client 
    for this experiment, a Redis instance is being used to stream position data, but this can be changed as needed
    ''' 
    with Actuator("/dev/cu.usbmodem21301") as actuator:
        with redis.Redis(host='localhost', port=6379, db=0) as ipc_client:
            try:
                Agent.set_actuator(actuator)
                Platform(ipc_client).control()
                
            except KeyboardInterrupt:
                actuator.stop_all()