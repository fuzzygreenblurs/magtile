import numpy as np
from agent_color import AgentColor
from agent import Agent
from platform import Platform
from actuator import Actuator

if __name__ == '__main__':
    with Actuator("/dev/cu.usbmodem21301") as actuator:
        platform = Platform()
        
        Agent.set_actuator(actuator)
        platform.add_disc(Agent(AgentColor.YELLOW))
        platform.add_disc(Agent(AgentColor.BLACK))

        try:
            platform.control()
        except KeyboardInterrupt:
            actuator.stop_all()