import redis
import numpy as np
from agent import Agent
from magtile_platform import Platform
from actuator import Actuator
 
if __name__ == '__main__':
    with Actuator("/dev/cu.usbmodem21301") as actuator:
        with redis.Redis(host='localhost', port=6379, db=0) as ipc_client:
            try:
                Agent.set_actuator(actuator)
                platform = Platform(ipc_client)
                platform.control()
                
            except KeyboardInterrupt:
                actuator.stop_all()

    # with Actuator("/dev/cu.usbmodem21301") as actuator:
    #     '''
    #     note: the term ipc_client is used to descibe a generic interprocess communication client 
    #     for this experiment, a Redis instance is being used to stream position data, but this can be changed as needed
    #     ''' 
    #     with redis.Redis(host='localhost', port=6379, db=0) as ipc_client:
    #         try:
    #             from platform import Platform
    #             # from agent import Agent
    #             # Agent.set_actuator(actuator)
    #             platform  = Platform(ipc_client)
    #             # platform.control()
                
    #         except KeyboardInterrupt:
    #             actuator.stop_all()