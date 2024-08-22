import time
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
            '''
                - for each iteration of the control loop:
                    - if all agents are outside the interference range with each other, follow the individual shortest path calculated for each agent
                    - if any agent falls within the interference range of another agent, calculate a new shortest path without any interference
            '''

            #TODO: this should only run for the duration of the experiment set by num_samples
            while True:
                platform.update_agent_positions()
                agent_positions = [agent.position for agent in platform.agents]
                
                if platform.no_interference(agent_positions):
                    [agent.advance() for agent in platform.agents]
                else:
                    for agent in platform.agents:
                        current_position_index = agent.find_closest_coil()[0]
                        # TODO: djikstra2()
                        shortest_path = platform.calc_shortest_path(current_position_index, agent.input_trajectory[agent.current_index])
                        agent.input_trajectory[agent.current_index:agent.current_index + len(shortest_path)] = shortest_path
                        agent.advance()

                # TODO: stop driven coils from this iteration
                actuator.stop_all_driven()
                time.sleep(0.1)

        except KeyboardInterrupt:
            actuator.stop_all()