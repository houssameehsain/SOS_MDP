import os
import cv2
import gym
import numpy as np
from math import floor
from statistics import mean

import networkx as nx
from networkx.algorithms.approximation import node_connectivity
from networkx.algorithms.efficiency_measures import global_efficiency
from networkx.algorithms.centrality import closeness_centrality, betweenness_centrality
from momepy import straightness_centrality
from shapely.geometry import Point, LineString, Polygon

from helpers.log import log_plot_data


# Beady Ring version 00 env
class BR_v0(gym.Env):

    """ 
    An OpenAI Gym environment based on the 'Beady Ring' model (Hillier & Hanson, 1984).

    The environemnt has two variants, one with full observation of the whole celular automata 
    world, and another with a user adjustable local observation where the agent gets information 
    about the state of the neighboring cells only. (default: local observation)

    """ 

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    def __init__(self, run, save_img_freq=1000, local=True, screen_size=500):
        super(BR_v0, self).__init__()

        self.local = local
        self.screen_width = screen_size
        self.screen_height = screen_size
        self.save_img_freq = save_img_freq

        self._carrier_color = 127.5
        self._house_color = 0
        self._street_color = 255
        self._max_row_len = 36  # length and width of the cellular automata world

        self.pad = 0
        if self.local:
            self._obs_size = 3  # length and width of observation window
            self.pad = int(floor(self._obs_size/2))
        
        self.max_world_row_len = int(self._max_row_len + 2*self.pad)
        
        # initial location
        self.R = int(floor(self.max_world_row_len/2))
        self.C = int(floor(self.max_world_row_len/2))
        
        self.x = self.R - self.pad
        self.y = self.C - self.pad
        
        self.cell = [self.x, self.y]
        self.adjacent_cells = [[self.x, self.y]]
        self.adj_cells = [[self.x, self.y]]
        
        # initial state
        self.state = np.full([self.max_world_row_len, self.max_world_row_len], self._carrier_color)

        # initial observation
        self.observation = self.get_obs() 

        self.action_space = gym.spaces.Discrete(2)
        if self.local:
            self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                    shape=(1, self._obs_size**2), 
                                                    dtype=np.uint8) 
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                    shape=(1, self._max_row_len, self._max_row_len), 
                                                    dtype=np.uint8) 
        
        self.run = run
        self.path = f"images/{self.run.id}/"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        self.stp = 0
        self.eps = 0
        self.cumul_rwd = 0

        self.plot_returns = []
        self.plot_eps_lengths = []

        self.isopen = True

    def step(self, _cell_state):
        assert self.action_space.contains(_cell_state), f"{_cell_state} is an invalid action"

        self.stp += 1
        # update state step
        self.state[self.R, self.C] = 255*_cell_state
        
        deleted = 0
        for i, a in enumerate(self.adj_cells):
            if a == self.cell:
                del self.adj_cells[i]
                deleted += 1
        if deleted == 0:
            raise RuntimeError("The current action cell must be deleted at each step.")
        
        adjacent = self.get_adjacent()
        
        # reward
        reward = 0

        if _cell_state == 0: # house cell
            reward += 1/(self._max_row_len**2) # density 

        self.cumul_rwd += reward # update episode cumulative reward
        
        if _cell_state == 1: # Add adjacent cells only if it's a street cell
            for item in adjacent:
                if item not in self.adjacent_cells:
                    self.adjacent_cells.append(item)
                    self.adj_cells.append(item)
        
        # done
        done = len(self.adj_cells) == 0

        if len(self.adj_cells) > 0:
            # next action location selection
            self.cell = self.adj_cells[0] # random.choice(self.adj_cells)
            
            self.x = self.cell[0]
            self.y = self.cell[1]

            self.R = self.x + self.pad
            self.C = self.y + self.pad
        
        # observation
        self.observation = self.get_obs()
        
        # save state image to project dir
        if self.eps == 1 or self.eps % self.save_img_freq == 0:
            frame = self.make_screen()
            cv2.imwrite(self.path + f"BR_v0_eps-{self.eps}_step-{self.stp}.png", frame)

        # log returns and episode lengths
        if done:
            self.plot_returns.append(self.cumul_rwd)
            self.plot_eps_lengths.append(self.stp)
        
        return self.observation, reward, done, {}

    def reset(self):
        # initial state
        self.state = np.full([self.max_world_row_len, self.max_world_row_len], self._carrier_color)
        
        # reset counters
        self.eps += 1
        self.stp = 0
        self.cumul_rwd = 0
        
        # initial location
        self.R = int(floor(self.max_world_row_len/2))
        self.C = int(floor(self.max_world_row_len/2))

        self.x = self.R - self.pad
        self.y = self.C - self.pad

        self.cell = [self.x, self.y]
        self.adjacent_cells = [[self.x, self.y]]
        self.adj_cells = [[self.x, self.y]]

        # initial observation
        self.observation = self.get_obs()
        
        return self.observation

    def render(self, mode='human'):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either 'human' or 'rgb_array'"

        frame = self.make_screen()

        if mode == "human":
            cv2.imshow('BeadyRing_v0', frame)
            cv2.waitKey(0)

        return frame if mode == "rgb_array" else self.isopen

    def get_obs(self):
        # observation
        if self.local:
            left_up_R = int(self.R - self.pad)
            left_up_C = int(self.C - self.pad)
            right_bottom_R = int(self.R + self.pad)
            right_bottom_C = int(self.C + self.pad)
            obs_ = self.state[left_up_R:right_bottom_R + 1, left_up_C:right_bottom_C + 1]
            obs_ = obs_.reshape(1, self._obs_size**2)
        else:
            obs_ = self.state
        
        return obs_

    def get_adjacent(self):
        # von Neumann neighborhood
        adjacent = []
        if self.x < self._max_row_len - 1:
            adjacent.append([self.x+1, self.y])
        if self.y > 0:
            adjacent.append([self.x, self.y-1])
        if self.x > 0:
            adjacent.append([self.x-1, self.y])
        if self.y < self._max_row_len - 1:
            adjacent.append([self.x, self.y+1])
        return adjacent
    
    def make_screen(self):
        if self.local:
            img = self.state[self.pad:self._max_row_len+self.pad, self.pad:self._max_row_len+self.pad]
        else:
            img = self.state
        img = img.reshape(self._max_row_len, self._max_row_len)
        img = cv2.resize(img, (self.screen_width, self.screen_height), interpolation = cv2.INTER_AREA)
        frame = np.zeros([self.screen_width, self.screen_height, 3], dtype=np.uint8)
        frame[:,:,0] = img.astype(np.uint8)
        frame[:,:,1] = img.astype(np.uint8)
        frame[:,:,2] = img.astype(np.uint8)

        # add eps, step, rwd text to frame
        frame = cv2.copyMakeBorder(frame, 50, 5, 5, 5, cv2.BORDER_CONSTANT, None, value = [255, 255, 255])
        total_return = float("{0:.3f}".format(self.cumul_rwd))
        frame = cv2.putText(frame, f"Episode {self.eps} Step {self.stp} Return {total_return}", 
                        (5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0, 255), 2)
        return frame
    
    def close(self): 
        log_plot_data(
            self.plot_returns,
            self.plot_eps_lengths,
            self.run
        )
        self.isopen = False


# Beady Ring version 01 env
class BR_v1(gym.Env):

    """ 
    An OpenAI Gym environment based on the 'Beady Ring' model (Hillier & Hanson, 1984) with 
    NetworkX graph representations of the street network.

    The environemnt has two variants, one with full observation of the whole celular automata 
    world, and another with a user adjustable local observation where the agent gets information 
    about the state of the neighboring cells only. (default: local observation)

    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    def __init__(self, run, save_img_freq=1000, local=True, screen_size=500):
        super(BR_v1, self).__init__()

        self.local = local
        self.screen_width = screen_size
        self.screen_height = screen_size
        self.save_img_freq = save_img_freq

        self._carrier_color = 127.5
        self._house_color = 0
        self._street_color = 255
        self._cell_size = 3
        self._max_row_len = 36  # length and width of the cellular automata world

        self.pad = 0
        if self.local:
            self._obs_size = 3  # length and width of observation window
            self.pad = int(floor(self._obs_size/2))
        
        self.max_world_row_len = int(self._max_row_len + 2*self.pad)
        
        # initial location
        self.R = int(floor(self.max_world_row_len/2))
        self.C = int(floor(self.max_world_row_len/2))
        
        self.x = self.R - self.pad
        self.y = self.C - self.pad
        
        self.cell = [self.x, self.y]
        self.adjacent_cells = [[self.x, self.y]]
        self.adj_cells = [[self.x, self.y]]

        # initial street graph
        self.graph = nx.Graph()

        # shapely grid world 
        self.grid = []
        for y in reversed(range(self._max_row_len)):
            poly_row = []
            for x in range(self._max_row_len):
                cell_poly = Polygon([(x, y), (x+1, y), (x+1, y+1), (x, y+1)])
                poly_row.append(cell_poly)
            self.grid.append(poly_row)

        # initial state
        self.state = np.full([self.max_world_row_len, self.max_world_row_len], self._carrier_color)

        # initial observation
        self.observation = self.get_obs() 

        self.action_space = gym.spaces.Discrete(2)
        if self.local:
            self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                    shape=(1, self._obs_size**2), 
                                                    dtype=np.uint8) 
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                    shape=(1, self._max_row_len, self._max_row_len), 
                                                    dtype=np.uint8) 
        
        self.run = run
        self.path = f"images/{self.run.id}/"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        self.stp = 0
        self.eps = 0
        self.cumul_rwd = 0

        self.plot_returns = []
        self.plot_eps_lengths = []

        self.isopen = True

    def step(self, _cell_state):
        assert self.action_space.contains(_cell_state), f"{_cell_state} is an invalid action"

        self.stp += 1
        # update state step
        self.state[self.R, self.C] = 255*_cell_state
        
        deleted = 0
        for i, a in enumerate(self.adj_cells):
            if a == self.cell:
                del self.adj_cells[i]
                deleted += 1
        if deleted == 0:
            raise RuntimeError("The current action cell must be deleted at each step.")
        
        adjacent = self.get_adjacent()
        
        if _cell_state == 1:
            # Add street node to graph
            point = self.grid[self.x][self.y].centroid
            self.graph.add_node((point.x, point.y)) 

            for item in adjacent:
                # Add adjacent cells only if they're street cells
                if item not in self.adjacent_cells: 
                    self.adjacent_cells.append(item)
                    self.adj_cells.append(item)

                # update street graph edges
                _adj_x = item[0]
                _adj_y = item[1]
                _adj_state = self.state[_adj_x+self.pad, _adj_y+self.pad]
                # Add street graph edge if adjacent cell is a street cell 
                if round(_adj_state/255) == 1:
                    _adj_point = self.grid[_adj_x][_adj_y].centroid
                    self.graph.add_edge((point.x, point.y), (_adj_point.x, _adj_point.y)) 

        # reward
        reward = 0

        # print(f'action {_cell_state}')
        if not nx.is_empty(self.graph): 
            reward += node_connectivity(self.graph)  # connectivity reward
            reward += global_efficiency(self.graph)  # efficiency reward 
            reward += mean(list(closeness_centrality(self.graph).values()))  # closeness reward
            reward += mean(list(betweenness_centrality(self.graph).values()))  # betweenness reward
            # reward += straightness_centrality(self.graph, normalized=True, verbose=False)

        if _cell_state == 0: # house cell
            reward += 1/(self._max_row_len**2)  # density 

        self.cumul_rwd += reward  # update episode cumulative reward

        # done
        done = len(self.adj_cells) == 0

        if len(self.adj_cells) > 0:
            # next action location selection
            self.cell = self.adj_cells[0]  # random.choice(self.adj_cells)
            
            self.x = self.cell[0]
            self.y = self.cell[1]

            self.R = self.x + self.pad
            self.C = self.y + self.pad
        
        # observation
        self.observation = self.get_obs()
        
        # save state image to project dir
        if self.eps == 1 or self.eps % self.save_img_freq == 0:
            frame = self.make_screen(show_graph=True)
            cv2.imwrite(self.path + f"BR_v1_eps-{self.eps}_step-{self.stp}.png", frame)

        # log returns and episode lengths
        if done:
            self.plot_returns.append(self.cumul_rwd)
            self.plot_eps_lengths.append(self.stp)
        
        return self.observation, reward, done, {}

    def reset(self):
        # initial state
        self.state = np.full([self.max_world_row_len, self.max_world_row_len], self._carrier_color)
        
        # reset counters
        self.eps += 1
        self.stp = 0
        self.cumul_rwd = 0
        
        # initial location
        self.R = int(floor(self.max_world_row_len/2))
        self.C = int(floor(self.max_world_row_len/2))

        self.x = self.R - self.pad
        self.y = self.C - self.pad

        self.cell = [self.x, self.y]
        self.adjacent_cells = [[self.x, self.y]]
        self.adj_cells = [[self.x, self.y]]

        # initial street graph
        self.graph = nx.Graph()

        # initial observation
        self.observation = self.get_obs()
        
        return self.observation

    def render(self, mode='human'):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either 'human' or 'rgb_array'"

        frame = self.make_screen(show_graph=True)

        if mode == "human":
            cv2.imshow('BeadyRing_v1', frame)
            cv2.waitKey(0)

        return frame if mode == "rgb_array" else self.isopen

    def get_obs(self):
        # observation
        if self.local:
            left_up_R = int(self.R - self.pad)
            left_up_C = int(self.C - self.pad)
            right_bottom_R = int(self.R + self.pad)
            right_bottom_C = int(self.C + self.pad)
            obs_ = self.state[left_up_R:right_bottom_R + 1, left_up_C:right_bottom_C + 1]
            obs_ = obs_.reshape(1, self._obs_size**2)
        else:
            obs_ = self.state
        
        return obs_

    def get_adjacent(self):
        # von Neumann neighborhood
        adjacent = []
        if self.x < self._max_row_len - 1:
            adjacent.append([self.x+1, self.y])
        if self.y > 0:
            adjacent.append([self.x, self.y-1])
        if self.x > 0:
            adjacent.append([self.x-1, self.y])
        if self.y < self._max_row_len - 1:
            adjacent.append([self.x, self.y+1])
        return adjacent

    def make_screen(self, show_graph=False):
        
        if self.local:
            img = self.state[self.pad:self._max_row_len+self.pad, self.pad:self._max_row_len+self.pad]
        else:
            img = self.state

        if show_graph:
            frame = np.zeros([self._max_row_len, self._max_row_len, 3], np.uint8)

            # draw shapely polygon cells on frame
            for y in reversed(range(self._max_row_len)):
                for x in range(self._max_row_len):
                    cell = self.grid[y][x]
                    coords = np.array(cell.exterior.coords, dtype=np.int32)
                    coords = coords.reshape((-1,1,2))
                    cell_state = img[y][x]
                    if round(cell_state / self._street_color) == 1:
                        cv2.fillPoly(frame, [coords], (255,255,255))
                    elif cell_state == self._house_color:
                        cv2.fillPoly(frame, [coords], (0,0,0))
                    else:
                        cv2.fillPoly(frame, [coords], (127,127,127))
                    
            # draw graph edges on frame
            for edge in self.graph.edges:
                point1 = edge[0]
                point2 = edge[1]
                cv2.line(frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0,0,255), 1)

            frame = cv2.resize(frame, (self.screen_width, self.screen_height), interpolation = cv2.INTER_AREA)

        else:
            img = img.reshape(self._max_row_len, self._max_row_len)
            img = cv2.resize(img, (self.screen_width, self.screen_height), interpolation = cv2.INTER_AREA)
            frame = np.zeros([self.screen_width, self.screen_height, 3], dtype=np.uint8)
            frame[:,:,0] = img.astype(np.uint8)
            frame[:,:,1] = img.astype(np.uint8)
            frame[:,:,2] = img.astype(np.uint8)

        # add eps, step, rwd text to frame 
        frame = cv2.copyMakeBorder(frame, 50, 5, 5, 5, cv2.BORDER_CONSTANT, None, value = [255, 255, 255])
        total_return = float("{0:.3f}".format(self.cumul_rwd))
        frame = cv2.putText(frame, f"Episode {self.eps} Step {self.stp} Return {total_return}", 
                        (5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0, 255), 2)
        return frame
    
    def close(self): 
        log_plot_data(
            self.plot_returns,
            self.plot_eps_lengths,
            self.run
        )
        self.isopen = False


