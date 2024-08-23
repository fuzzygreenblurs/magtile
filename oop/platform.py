import json
import time
import math
import threading
import numpy as np
import networkx as nx
from agent import Agent
from agent_color import AgentColor
from constants import *
'''
all values are measured in centimeters [cm] unless inches [in] are explicitly specified in variable name
variable notation: "x" represents a value in centimeters. "x_inches" represents a value in inches
'''

class Platform:
    def __init__(self, ipc_client):
        self.ipc_client = ipc_client
        self.create_agents()

    def control(self):
        '''
            for each iteration of the control loop:
                - if all agents are outside the interference range with each other, follow the individual shortest path calculated for each agent
                - if any agent falls within the interference range of another agent, calculate a new shortest path without any interference
        '''

        for i in NUM_SAMPLES:
            self.update_all_agent_positions()
            if self.any_agents_inside_interference_zone():
                # TODO: shortest_paths = multi_agent_dijkstra()
                # TODO: [agent.update_motion_plan({steps: 2, shortest_paths) for path in shortest_paths]
                pass

            # [agent.advance(i) for agent in self.agents]
            self.advance_agents(i)

    def advance_agents(self, i):
        threads = []
        for agent in self.agents:
            thread = threading.Thread(target=agent.advance, args=(i,))
            threads.append(thread)

        [thread.start() for thread in threads]
        [thread.join() for thread in threads]

    def create_agents(self):
        '''
            - placeholder function to instantiate the target agents
            - TODO: this should be replaced with a more generic add_agent function instead 
        '''
        yellow = Agent(self, AgentColor.YELLOW)
        black  = Agent(self, AgentColor.BLACK)
        self.agents = [yellow, black]

    def update_all_agent_positions(self):
        messages = self.ipc_client.xrevrange(POSITIONS_STREAM, count=1)
        
        if messages:
            _, message = messages[0]
            for agent in self.agents:
                color = f"b\'{agent.color}\'".encode('utf-8')
                agent.update_position(json.loads(message[color].decode()))

        else:
            return None

    def update_adjacency_for_collision_avoidance(self, path, threshold):
        '''
        Modify the adjacency matrix to avoid paths that are too close to a given path.
        '''

        A = self.adjacency_matrix.copy()
        for i in range(len(path)):
            for j in range(len(A)):
                distance = np.linalg.norm(np.array(self.grid_positions[path[i]]) - np.array(self.grid_positions[j]))
                if distance < threshold:
                    A[path[i], j] = np.inf
                    A[j, path[i]] = np.inf
        return A

    def multi_agent_dijkstra(self, start_idx, end_idx, adjacency_matrix=None):
        '''
            - for each agent, calculate the shortest path from its current position to its target
            - djikstra_2 should be able to handle k agents and enforce the interference zone restriction for all of them 
        '''
        if adjacency_matrix is None:
            adjacency_matrix = self.adjacency_matrix
        graph = nx.from_numpy_array(adjacency_matrix)
        shortest_path = nx.dijkstra_path(graph, start_idx, end_idx)
        return shortest_path

    def any_agents_inside_interference_zone(self):
        '''
            for an arbitrary number of agents, check if each possible pair of agents is within the interference range
        '''

        agent_positions = [a.position for a in self.agents]

        for i in range(len(agent_positions)):
            for j in range(i + 1, len(agent_positions)):
                if np.linalg.norm(agent_positions[i] - agent_positions[j]) <= FIELD_RANGE:
                    return True
        return False