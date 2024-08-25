import asyncio
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
                asyncio.run(platform.control())
                
            except KeyboardInterrupt:
                actuator.stop_all()