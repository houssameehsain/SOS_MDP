import os
import gym
import numpy as np
from math import floor, sin, cos, radians
from statistics import mean
from scipy import interpolate 
from scipy.ndimage import label
import itertools
import noise
import random

import networkx as nx

from matplotlib import use
import matplotlib.pyplot as plt 
use("Agg")
from matplotlib.patches import Polygon 


# Beady Ring version 00 env
class GBR_v0(gym.Env):

    """ 
    An OpenAI Gym environment based on the dynamics of the 'Beady Ring' model (Hillier & Hanson, 1984).

    The environemnt has two variants, one with full observation of the whole celular automata 
    world, and another with a user adjustable local observation where the agent gets information 
    about the state of the neighboring cells only. (default: local observation)

    """ 

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    def __init__(self, local=True, screen_size=500):
        super(GBR_v0, self).__init__()

        self.local = local
        self.screen_width = screen_size
        self.screen_height = screen_size

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

        # grid world 
        x, y = np.meshgrid(np.arange(self._max_row_len), np.arange(self._max_row_len))
        cell_poly = np.stack([x, y, x+1, y, x+1, y+1, x, y+1], axis=-1)
        self.grid = np.flip(cell_poly.reshape(self._max_row_len, self._max_row_len, 4, 2), axis=1).tolist()

        # initial state
        self.state = np.full((self.max_world_row_len, self.max_world_row_len), self._carrier_color, dtype=np.uint8)

        self.action_space = gym.spaces.Discrete(2)
        if self.local:
            self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                    shape=(self._obs_size**2,), 
                                                    dtype=np.uint8) 
        else: 
            self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                    shape=(1, self._max_row_len, self._max_row_len), 
                                                    dtype=np.uint8) 

        self.stp = 0
        self.eps = 0
        self.cumul_rwd = 0.0
        self.cumul_perf = 0.0

        self.returns = []
        self.performances = []
        self.eps_lengths = []

        self.isopen = True

    def reset(self):
        # initial state
        self.state = np.full((self.max_world_row_len, self.max_world_row_len), self._carrier_color, dtype=np.uint8)
        
        # reset counters
        self.eps += 1
        self.stp = 0
        self.cumul_rwd = 0.0
        self.cumul_perf = 0.0
        
        # initial location
        self.R = int(floor(self.max_world_row_len/2))
        self.C = int(floor(self.max_world_row_len/2))

        self.x = self.R - self.pad
        self.y = self.C - self.pad

        self.cell = [self.x, self.y]
        self.adjacent_cells = [[self.x, self.y]]
        self.adj_cells = [[self.x, self.y]]

        # initial observation
        observation = self.get_obs()

        self.returns.append(self.cumul_rwd)
        self.performances.append(self.cumul_perf)
        self.eps_lengths.append(self.stp + 1)
        
        return observation
    
    def step(self, _cell_state):
        assert self.action_space.contains(_cell_state), f"{_cell_state} is an invalid action"

        self.stp += 1
        # update state step
        self.state[self.R, self.C] = 255*_cell_state
        
        original_length = len(self.adj_cells)
        del self.adj_cells[0]
        deleted = len(self.adj_cells) - original_length
        if deleted == 0:
            raise RuntimeError("The current action cell must be deleted at each step.")
        
        # reward
        rwd = 0.0
        perf = 0.0
        if _cell_state == 0: # house cell
            rwd += 1.0 
            perf += 1/(self._max_row_len**2) # density

        self.cumul_rwd += rwd # update episode cumulative reward
        self.cumul_perf += perf
        
        if _cell_state == 1: # Add adjacent cells only if it's a street cell
            # get von Neumann neighborhood
            adjacent = self.get_adjacent()

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
        observation = self.get_obs()

        # log returns and episode lengths
        if done:
            self.returns[-1] = self.cumul_rwd
            self.performances[-1] = self.cumul_perf
            self.eps_lengths[-1] = self.stp
        
        return observation, rwd, done, {}

    def render(self, mode="rgb_array"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either 'human' or 'rgb_array'"

        frame = self.make_screen()

        if mode == "human":
            # cv2.imshow('BeadyRing_v0', frame)
            # cv2.waitKey(0)
            pass

        return frame if mode == "rgb_array" else self.isopen

    def get_obs(self):
        # observation
        if self.local:
            left_up_R = int(self.R - self.pad)
            left_up_C = int(self.C - self.pad)
            right_bottom_R = int(self.R + self.pad)
            right_bottom_C = int(self.C + self.pad)
            obs_ = self.state[left_up_R:right_bottom_R + 1, left_up_C:right_bottom_C + 1]
            obs_ = obs_.reshape(self._obs_size**2,)
        else:
            obs_ = self.state.copy().reshape(1, self._max_row_len, self._max_row_len)

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
    
    def plt2arr(self, fig):
        fig.canvas.draw()
        # grab the pixel buffer and dump it into a numpy array
        rgb_array = np.array(fig.canvas.renderer._renderer)
        return rgb_array
    
    def make_screen(self):
        if self.local:
            img = self.state[self.pad:self._max_row_len+self.pad, self.pad:self._max_row_len+self.pad]
        else:
            img = self.state.copy()

        # Create a figure and axis object
        fig, ax = plt.subplots()

        # Create cell patches and add them to the axis
        for y in reversed(range(self._max_row_len)):
            for x in range(self._max_row_len):
                cell = self.grid[y][x]
                cell_state = img[y][x]
                if round(cell_state / self._street_color) == 1:
                    poly_patch = Polygon(np.array(cell), facecolor='white', edgecolor='gray')
                elif cell_state == self._house_color:
                    poly_patch = Polygon(np.array(cell), facecolor='black', edgecolor='gray')
                else:
                    poly_patch = Polygon(np.array(cell), facecolor='gray', edgecolor='gray')
                ax.add_patch(poly_patch)

        # plt.imshow(frame)

        ax.autoscale()
        ax.set_aspect("equal")

        # add eps, step, rwd text to frame
        total_return = float("{0:.3f}".format(self.cumul_rwd))
        plt.title(f"Episode {self.eps} Step {self.stp} Return {total_return}", fontsize=14, loc='center', fontweight='normal')

        # np array from plot
        frame = self.plt2arr(fig)

        plt.close()
        return frame

    def close(self): 
        self.isopen = False

# Beady Ring version 01 env
class GBR_v1(gym.Env):

    """ 
    An OpenAI Gym environment based on the dynamics of the 'Beady Ring' model (Hillier & Hanson, 1984).

    The environemnt has two variants, one with full observation of the whole celular automata 
    world, and another with a user adjustable local observation where the agent gets information 
    about the state of the neighboring cells only. (default: local observation)

    """ 

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    def __init__(self, local=True, screen_size=500):
        super(GBR_v1, self).__init__()

        self.local = local
        self.screen_width = screen_size
        self.screen_height = screen_size

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

        # grid world 
        x, y = np.meshgrid(np.arange(self._max_row_len), np.arange(self._max_row_len))
        cell_poly = np.stack([x, y, x+1, y, x+1, y+1, x, y+1], axis=-1)
        self.grid = np.flip(cell_poly.reshape(self._max_row_len, self._max_row_len, 4, 2), axis=1).tolist()

        # initial state
        self.state = np.full((self.max_world_row_len, self.max_world_row_len), self._carrier_color, dtype=np.uint8)

        self.action_space = gym.spaces.Discrete(2)
        if self.local:
            self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                    shape=(1, self._obs_size**2), 
                                                    dtype=np.uint8) 
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                    shape=(1, self._max_row_len, self._max_row_len), 
                                                    dtype=np.uint8) 
        
        self.stp = 0
        self.eps = 0
        self.cumul_rwd = 0.0
        self.cumul_perf = 0.0
        self.settlement_size = 0.0
        self.block_perf = 0.0
        self.prev_perf = 0.0

        self.returns = []
        self.performances = []
        self.eps_lengths = []

        self.isopen = True

    def reset(self):
        # initial state
        self.state = np.full((self.max_world_row_len, self.max_world_row_len), self._carrier_color, dtype=np.uint8)
        
        # reset counters
        self.eps += 1
        self.stp = 0
        self.cumul_rwd = 0.0
        self.cumul_perf = 0.0
        
        # initial location
        self.R = int(floor(self.max_world_row_len/2))
        self.C = int(floor(self.max_world_row_len/2))

        self.x = self.R - self.pad
        self.y = self.C - self.pad

        self.cell = [self.x, self.y]
        self.adjacent_cells = [[self.x, self.y]]
        self.adj_cells = [[self.x, self.y]]

        # initial observation
        observation = self.get_obs()
        self.settlement_size = 0.0
        self.block_perf = 0.0
        self.prev_perf = 0.0

        self.returns.append(self.cumul_rwd)
        self.performances.append(self.cumul_perf)
        self.eps_lengths.append(self.stp + 1)
        
        return observation
    
    def step(self, _cell_state):
        assert self.action_space.contains(_cell_state), f"{_cell_state} is an invalid action"

        self.stp += 1
        # update state step
        self.state[self.R, self.C] = 255*_cell_state
        
        original_length = len(self.adj_cells)
        del self.adj_cells[0]
        deleted = len(self.adj_cells) - original_length
        if deleted == 0:
            raise RuntimeError("The current action cell must be deleted at each step.")
        
        if _cell_state == 1: # street cell
            # get von Neumann neighborhood
            adjacent = self.get_adjacent()

            for item in adjacent:
                # Add adjacent cells only if they're street cells
                if item not in self.adjacent_cells: 
                    self.adjacent_cells.append(item)
                    self.adj_cells.append(item)

        # performance
        if _cell_state == 0: # house cell
            self.settlement_size += 1.0

            # urban block performance
            self.block_perf = 0.0
            block_mask = self.state.copy()
            block_mask[block_mask == 0] = 1
            block_mask[block_mask != 1] = 0
            labeled_arr, num_blocks = label(block_mask)
            block_sizes = np.bincount(labeled_arr.ravel())
            block_sizes = block_sizes[1:]
            for size in block_sizes:
                # print("block detected")
                if size >= 6:
                    self.block_perf += 10.0

        perf = self.settlement_size + self.block_perf

        rwd = perf - self.prev_perf
        
        self.cumul_rwd += rwd  # update episode cumulative reward
        self.cumul_perf = perf
        self.prev_perf = perf # update previous state performance

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
        observation = self.get_obs()

        # log returns and episode lengths
        if done:
            self.returns[-1] = self.cumul_rwd
            self.performances[-1] = self.cumul_perf
            self.eps_lengths[-1] = self.stp
        
        return observation, rwd, done, {}

    def render(self, mode="rgb_array"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either 'human' or 'rgb_array'"

        frame = self.make_screen()

        if mode == "human":
            # cv2.imshow('BeadyRing_v1', frame)
            # cv2.waitKey(0)
            pass

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
            obs_ = self.state.copy().reshape(1, self._max_row_len, self._max_row_len)
        
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

    def plt2arr(self, fig):
        fig.canvas.draw()
        # grab the pixel buffer and dump it into a numpy array
        rgb_array = np.array(fig.canvas.renderer._renderer)
        return rgb_array
    
    def make_screen(self):
        if self.local:
            img = self.state[self.pad:self._max_row_len+self.pad, self.pad:self._max_row_len+self.pad]
        else:
            img = self.state.copy()

        # Create a figure and axis object
        fig, ax = plt.subplots()

        # Create cell patches and add them to the axis
        for y in reversed(range(self._max_row_len)):
            for x in range(self._max_row_len):
                cell = self.grid[y][x]
                cell_state = img[y][x]
                if round(cell_state / self._street_color) == 1:
                    poly_patch = Polygon(np.array(cell), facecolor='white', edgecolor='gray')
                elif cell_state == self._house_color:
                    poly_patch = Polygon(np.array(cell), facecolor='black', edgecolor='gray')
                else:
                    poly_patch = Polygon(np.array(cell), facecolor='gray', edgecolor='gray')
                ax.add_patch(poly_patch)

        # plt.imshow(frame)

        ax.autoscale()
        ax.set_aspect("equal")

        # add eps, step, rwd text to frame
        total_return = float("{0:.3f}".format(self.cumul_rwd))
        plt.title(f"Episode {self.eps} Step {self.stp} Return {total_return}", fontsize=14, loc='center', fontweight='normal')

        # np array from plot
        frame = self.plt2arr(fig)

        plt.close()
        return frame

    def close(self): 
        self.isopen = False

# Beady Ring version 02 env
class GBR_v2(gym.Env):

    """ 
    An OpenAI Gym environment based on the dynamics of the 'Beady Ring' model (Hillier & Hanson, 1984) with 
    NetworkX graph representations of the street network.

    The environemnt has two variants, one with full observation of the whole celular automata 
    world, and another with a user adjustable local observation where the agent gets information 
    about the state of the neighboring cells only. (default: local observation)

    """ 

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    def __init__(self, local=True, screen_size=500):
        super(GBR_v2, self).__init__()

        self.local = local
        self.screen_width = screen_size
        self.screen_height = screen_size

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

        # grid world 
        x, y = np.meshgrid(np.arange(self._max_row_len), np.arange(self._max_row_len))
        cell_poly = np.stack([x, y, x+1, y, x+1, y+1, x, y+1], axis=-1)
        self.grid = np.flip(cell_poly.reshape(self._max_row_len, self._max_row_len, 4, 2), axis=1).tolist()

        # initial state
        self.state = np.full((self.max_world_row_len, self.max_world_row_len), self._carrier_color, dtype=np.uint8)

        self.action_space = gym.spaces.Discrete(2)
        if self.local:
            self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                    shape=(1, self._obs_size**2), 
                                                    dtype=np.uint8) 
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                    shape=(1, self._max_row_len, self._max_row_len), 
                                                    dtype=np.uint8) 
        
        self.stp = 0
        self.eps = 0
        self.cumul_rwd = 0.0
        self.cumul_perf = 0.0
        self.settlement_size = 0.0

        self.returns = []
        self.performances = []
        self.eps_lengths = []

        self.isopen = True

    def reset(self):
        # initial state
        self.state = np.full((self.max_world_row_len, self.max_world_row_len), self._carrier_color, dtype=np.uint8)
        
        # reset counters
        self.eps += 1
        self.stp = 0
        self.cumul_rwd = 0.0
        self.cumul_perf = 0.0
        
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
        observation = self.get_obs()
        self.settlement_size = 0.0

        self.returns.append(self.cumul_rwd)
        self.performances.append(self.cumul_perf)
        self.eps_lengths.append(self.stp + 1)
        
        return observation
    
    def step(self, _cell_state):
        assert self.action_space.contains(_cell_state), f"{_cell_state} is an invalid action"

        self.stp += 1
        # update state step
        self.state[self.R, self.C] = 255*_cell_state
        
        original_length = len(self.adj_cells)
        del self.adj_cells[0]
        deleted = len(self.adj_cells) - original_length
        if deleted == 0:
            raise RuntimeError("The current action cell must be deleted at each step.")
        
        if _cell_state == 1: # street cell
            # get von Neumann neighborhood
            adjacent = self.get_adjacent()

            # Add street node to graph
            point = self.get_centroid(self.grid[self.x][self.y])
            self.graph.add_node(point, pos=point) 

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
                    _adj_point = self.get_centroid(self.grid[_adj_x][_adj_y])
                    self.graph.add_node(_adj_point, pos=_adj_point)
                    self.graph.add_edge(point, _adj_point) 

        rwd = 0.0
        # performance
        if _cell_state == 0: # house cell
            self.settlement_size += 1.0
            rwd += 1.0

        self.cumul_rwd += rwd  # update episode cumulative reward

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
        observation = self.get_obs()

        # log returns and episode lengths
        if done:
            # street network performance
            cycle_perf = 0.0 
            try:
                cycles = nx.minimum_cycle_basis(self.graph)
                n_cycles = len(cycles) 
                # continue only if an additional cycle is detected 
                if n_cycles > 0: 
                    # get lengths of all cycles 
                    cycle_lengths = [len(x) for x in cycles]
                    for i, length in enumerate(cycle_lengths):
                        if length >= 12:
                            cycle_perf += 10.0
            except nx.exception.NetworkXNoCycle as e:
                pass

            perf = self.settlement_size + cycle_perf
            self.cumul_perf = perf

            rwd += cycle_perf
            self.cumul_rwd += cycle_perf  # update episode cumulative reward

            self.returns[-1] = self.cumul_rwd
            self.performances[-1] = self.cumul_perf
            self.eps_lengths[-1] = self.stp
        
        return observation, rwd, done, {}

    def render(self, mode="rgb_array"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either 'human' or 'rgb_array'"

        frame = self.make_screen(show_graph=True)

        if mode == "human":
            # cv2.imshow('BeadyRing_v1', frame)
            # cv2.waitKey(0)
            pass

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
            obs_ = self.state.copy().reshape(1, self._max_row_len, self._max_row_len)
        
        return obs_

    def get_centroid(self, vertices):
        x_list = [vertex[0] for vertex in vertices]
        y_list = [vertex[1] for vertex in vertices]
        n_pts = len(vertices)
        x = sum(x_list) / n_pts
        y = sum(y_list) / n_pts
        return (x, y)
    
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

    def plt2arr(self, fig):
        fig.canvas.draw()
        # grab the pixel buffer and dump it into a numpy array
        rgb_array = np.array(fig.canvas.renderer._renderer)
        return rgb_array
    
    def make_screen(self, show_graph=False):
        if self.local:
            img = self.state[self.pad:self._max_row_len+self.pad, self.pad:self._max_row_len+self.pad]
        else:
            img = self.state.copy()

        # Create a figure and axis object
        fig, ax = plt.subplots()

        # Create cell patches and add them to the axis
        for y in reversed(range(self._max_row_len)):
            for x in range(self._max_row_len):
                cell = self.grid[y][x]
                cell_state = img[y][x]
                if round(cell_state / self._street_color) == 1:
                    poly_patch = Polygon(np.array(cell), facecolor='white', edgecolor='gray')
                elif cell_state == self._house_color:
                    poly_patch = Polygon(np.array(cell), facecolor='black', edgecolor='gray')
                else:
                    poly_patch = Polygon(np.array(cell), facecolor='gray', edgecolor='gray')
                ax.add_patch(poly_patch)

        if show_graph:
            # draw graph edges on frame
            nx.draw(self.graph, nx.get_node_attributes(self.graph, 'pos'), with_labels=False, node_size=0, width=1.0, edge_color='b', ax=ax)

        # plt.imshow(frame)

        ax.autoscale()
        ax.set_aspect("equal")

        # add eps, step, rwd text to frame
        total_return = float("{0:.3f}".format(self.cumul_rwd))
        plt.title(f"Episode {self.eps} Step {self.stp} Return {total_return}", fontsize=14, loc='center', fontweight='normal')

        # np array from plot
        frame = self.plt2arr(fig)

        plt.close()
        return frame

    def close(self): 
        self.isopen = False


# Path Node version 00 env
class GPN_v0(gym.Env):

    """ 
    An OpenAI Gym environment based on the dynamics of the 'Path Node' model (Erickson & Lloyd-Jones, 1997) with 
    topographic constraints.

    """ 

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    def __init__(self, screen_size=500):
        super(GPN_v0, self).__init__()

        self.screen_width = screen_size
        self.screen_height = screen_size

        self._carrier_color = 0.5
        self._house_color = 0
        self._street_color = 1
        self._cell_size = 3
        self.obs_size = 10
        self.max_slope = 0.3

        # define the world boundary 
        self.min_coord, self.max_coord = 10, 90  
        self.world_boundary = [
            [self.min_coord, self.min_coord],
            [self.max_coord, self.min_coord],
            [self.max_coord, self.max_coord],
            [self.min_coord, self.max_coord]
        ]
        
        # initialize terrain 
        self.terrain_seed = 123  # seed for terrain reproducibility 
        self.width = 100  # Width of the terrain array
        self.height = 100  # Height of the terrain array
        self.scale = 50  # Scale factor for the terrain heights
        octaves = 2  # Number of noise octaves
        persistence = 0.4  # Persistence parameter for noise
        lacunarity = 1.0  # Lacunarity parameter for noise
        central_flatness = 0.5 # Flatness factor for the central area

        self.terrain_elevations = self.generate_terrain(self.width, self.height, self.scale, 
                                                   octaves, persistence, lacunarity, 
                                                   self.terrain_seed, central_flatness)
        
        # Calculate the slope using the gradient of the elevations
        dx = np.gradient(self.terrain_elevations, axis=1)
        dy = np.gradient(self.terrain_elevations, axis=0)
        self.slopes = np.sqrt(dx**2 + dy**2)

        min_slope = np.min(self.slopes)
        max_slope = np.max(self.slopes)
        self.slopes = (self.slopes - min_slope) / (max_slope - min_slope)

        # initial cell geometry
        self.centroids = []
        self.cell_geometries = [] 
        self.tobe_cell_geometries = []
        self.house_geometries = []
        self.street_geometries = []

        # initial location
        x, y = 50.0, 40.0
        
        self.cell = [x, y]
        self.adjacent_cells = [[x, y]]
        self.adj_cells = [[x, y]]

        # initial state
        self.cell_states = [] 
        self.cell_slopes = []

        # initial street graph
        self.graph = nx.Graph() 

        self.encode_coord = interpolate.interp1d(
            [self.min_coord - 10, self.max_coord + 10],
            [0, 1]
        )

        move_space = [
            [0, 1],
            [-10, -5, 0, 5, 10]
        ]
        self.possible_moves = list(itertools.product(*move_space))

        # define action and observation spacesindex 
        self.action_space = gym.spaces.Discrete(len(self.possible_moves))
        self.observation_space = gym.spaces.Box(low=0, high=1, 
                                                shape=(4*self.obs_size,), 
                                                dtype=np.float32) 

        self.stp = 0
        self.eps = 0
        self.cumul_rwd = 0.0
        self.cumul_perf = 0.0
        self.settlement_size = 0.0

        self.returns = []
        self.performances = []
        self.eps_lengths = []

        self.isopen = True

    def reset(self):
        # initial state
        self.cell_states = []
        self.cell_slopes = []
        
        # reset counters
        self.eps += 1
        self.stp = 0
        self.cumul_rwd = 0.0
        self.cumul_perf = 0.0
        
        # initial location
        x, y = 50.0, 40.0

        self.cell = [x, y]
        self.adjacent_cells = [[x, y]]
        self.adj_cells = [[x, y]]

        # initial cell geometry
        self.centroids = []
        self.cell_geometries = [] 
        self.cell_geometry = self.make_cell_geometry(x, y, self._cell_size)
        self.tobe_cell_geometries = [self.cell_geometry.copy()]
        self.house_geometries = []
        self.street_geometries = []

        # initial street graph
        self.graph = nx.Graph()

        # initial observation
        observation = self.get_obs()
        self.settlement_size = 0.0

        self.returns.append(self.cumul_rwd)
        self.performances.append(self.cumul_perf)
        self.eps_lengths.append(self.stp + 1)
        
        return observation
    
    def step(self, action):
        self.stp += 1

        move = self.possible_moves[action]
        _cell_state = move[0]
        orientation = move[1]

        # rotate current cell geometry
        current_cell_geometry = self.cell_geometry.copy()
        if orientation != 0:
            current_cell_geometry = self.rotate_cell(current_cell_geometry, orientation)

        # get current terrain slope
        current_slope = self.slopes[floor(self.cell[1]), floor(self.cell[0])]

        # update state step
        self.cell_states.append(_cell_state)
        self.centroids.append(self.cell)
        self.cell_geometries.append(current_cell_geometry)
        self.cell_slopes.append(current_slope)

        if _cell_state == 0:
            self.house_geometries.append(current_cell_geometry)
        else:
            self.street_geometries.append(current_cell_geometry)
        
        original_length = len(self.adj_cells)
        del self.adj_cells[0]
        del self.tobe_cell_geometries[0]
        deleted = len(self.adj_cells) - original_length
        if deleted == 0:
            raise RuntimeError("The current action cell must be deleted at each step.")
        
        terminal_slope = False
        if _cell_state == 1:
            # get von Neumann neighborhood
            adjacent, adj_geom = self.get_adjacent(current_cell_geometry)

            # Add street node to graph
            node = (self.cell[0], self.cell[1])
            self.graph.add_node(node, pos=node) 

            for i, item in enumerate(adjacent):
                inside_tobes, _ = self.is_pt_in_geomList(item, self.tobe_cell_geometries)
                if inside_tobes:
                    continue
                else:
                    inside, cell_id = self.is_pt_in_geomList(item, self.cell_geometries)
                    if inside:
                        cell_type = self.cell_states[cell_id]
                        if cell_type == 0:
                            continue
                        else:
                            _adj_point = self.centroids[cell_id]
                            adj_node = (_adj_point[0], _adj_point[1])
                            self.graph.add_node(adj_node, pos=adj_node)
                            self.graph.add_edge(node, adj_node) 
                    else:
                        self.adjacent_cells.append(item)
                        self.adj_cells.append(item)
                        self.tobe_cell_geometries.append(adj_geom[i])
            
            terminal_slope = current_slope > self.max_slope

        rwd = 0.0
        # performance
        if _cell_state == 0: # house cell
            self.settlement_size += 1.0
            rwd += 1.0

        self.cumul_rwd += rwd  # update episode cumulative reward
        self.cumul_perf += rwd

        # done
        done = len(self.adj_cells) == 0 or self.is_outside_boundary(self.cell_geometry) or terminal_slope

        if len(self.adj_cells) > 0:
            # next action location selection
            self.cell = self.adj_cells[0]  # random.choice(self.adj_cells)
            self.cell_geometry = self.tobe_cell_geometries[0]
        
        # observation
        observation = self.get_obs()

        # log returns and episode lengths
        if done:
            self.returns[-1] = self.cumul_rwd
            self.performances[-1] = self.cumul_perf
            self.eps_lengths[-1] = self.stp
        
        return observation, rwd, done, {}

    def render(self, mode="rgb_array"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either 'human' or 'rgb_array'"

        frame = self.make_screen(show_graph=True)

        if mode == "human":
            # cv2.imshow('BeadyRing_v1', frame)
            # cv2.waitKey(0)
            pass

        return frame if mode == "rgb_array" else self.isopen

    def get_obs(self):
        # observation
        obs_ = []
        coord_fill = []
        slope_obs_ = []

        # get cell states
        obs_ += self.cell_states.copy()[-self.obs_size:]
        # get terrain slopes
        slope_obs_ += self.cell_slopes.copy()[-self.obs_size:]

        if len(obs_) < self.obs_size:
            coord_fill = [0]*(self.obs_size - len(obs_))
            obs_ += [0.5]*(self.obs_size - len(obs_))
            slope_obs_ += [0]*(self.obs_size - len(slope_obs_))
            
        # get cell coordinates
        xs = [pt[0] for pt in self.centroids]
        ys = [pt[1] for pt in self.centroids]
        # remap coordinates to [0, 1] range
        xs = self.encode_coord(xs).tolist() + coord_fill
        ys = self.encode_coord(ys).tolist() + coord_fill

        observation = list(itertools.chain(*zip(obs_, xs, ys, slope_obs_)))
        observation = np.array(observation, dtype=np.float32).reshape(4*self.obs_size,)
        return observation
    
    def get_adjacent(self, square):
        # von Neumann neighborhood
        side_length = self._cell_size

        # Calculate the direction vector of one of the sides of the square
        dx = square[1][0] - square[0][0]
        dy = square[1][1] - square[0][1]
        side_vector = [dx / side_length, dy / side_length]

        # Calculate the normal vector of the side vector
        normal_vector = [-side_vector[1], side_vector[0]]

        # Calculate the coordinates of the adjacent squares
        top_square = [[square[0][0] - normal_vector[0] * side_length, square[0][1] - normal_vector[1] * side_length],
                    [square[1][0] - normal_vector[0] * side_length, square[1][1] - normal_vector[1] * side_length],
                    [square[2][0] - normal_vector[0] * side_length, square[2][1] - normal_vector[1] * side_length],
                    [square[3][0] - normal_vector[0] * side_length, square[3][1] - normal_vector[1] * side_length]]
        top_center = self.center_of_square(top_square)

        right_square = [[square[1][0] + side_vector[0] * side_length, square[1][1] + side_vector[1] * side_length],
                        [square[2][0] + side_vector[0] * side_length, square[2][1] + side_vector[1] * side_length],
                        [square[3][0] + side_vector[0] * side_length, square[3][1] + side_vector[1] * side_length],
                        [square[0][0] + side_vector[0] * side_length, square[0][1] + side_vector[1] * side_length]]
        right_center = self.center_of_square(right_square)

        bottom_square = [[square[0][0] + normal_vector[0] * side_length, square[0][1] + normal_vector[1] * side_length],
                        [square[1][0] + normal_vector[0] * side_length, square[1][1] + normal_vector[1] * side_length],
                        [square[2][0] + normal_vector[0] * side_length, square[2][1] + normal_vector[1] * side_length],
                        [square[3][0] + normal_vector[0] * side_length, square[3][1] + normal_vector[1] * side_length]]
        bottom_center = self.center_of_square(bottom_square)

        left_square = [[square[3][0] - side_vector[0] * side_length, square[3][1] - side_vector[1] * side_length],
                    [square[0][0] - side_vector[0] * side_length, square[0][1] - side_vector[1] * side_length],
                    [square[1][0] - side_vector[0] * side_length, square[1][1] - side_vector[1] * side_length],
                    [square[2][0] - side_vector[0] * side_length, square[2][1] - side_vector[1] * side_length]]
        left_center = self.center_of_square(left_square)
        
        return [top_center, right_center, bottom_center, left_center], [top_square, right_square, bottom_square, left_square]
    
    def make_cell_geometry(self, x, y, side_len):
        # Calculate half the length of the square's side
        half_side_len = side_len / 2
        
        # Calculate the x and y coordinates of each corner relative to the center point
        top_left = [x - half_side_len, y + half_side_len]
        top_right = [x + half_side_len, y + half_side_len]
        bottom_left = [x - half_side_len, y - half_side_len]
        bottom_right = [x + half_side_len, y - half_side_len]
        
        # Return the coordinates of each corner as a nested list
        return [bottom_left, bottom_right, top_right, top_left]
    
    def rotate_cell(self, coords, angle):
        # Calculate the center of the square
        x_center = self.cell[0]
        y_center = self.cell[1]

        # Convert the angle to radians
        theta = radians(angle)

        # Rotate each corner of the square around the center
        rotated_coords = []
        for coord in coords:
            x = coord[0] - x_center
            y = coord[1] - y_center
            x_new = x * cos(theta) - y * sin(theta)
            y_new = x * sin(theta) + y * cos(theta)
            rotated_coords.append([x_new + x_center, y_new + y_center])

        return rotated_coords
    
    def center_of_square(self, coords):
        # Calculate the center point of the square using the average of the x and y coordinates of the corners
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        center_x = (max(x_coords) + min(x_coords)) / 2
        center_y = (max(y_coords) + min(y_coords)) / 2
        # Return the coordinates of the center point as a tuple
        return [center_x, center_y]
    
    def is_pt_in_geomList(self, pt, squares):
        x, y = pt[0], pt[1]

        for idx, square_corners in enumerate(squares):
            # Define the edges of the square
            edges = []
            for i in range(len(square_corners)):
                if i == len(square_corners)-1:
                    edges.append([square_corners[i], square_corners[0]])
                else:
                    edges.append([square_corners[i], square_corners[i+1]])

            # Check if the point is inside the square
            intersections = 0
            for edge in edges:
                if ((edge[0][1] > y) != (edge[1][1] > y)) and \
                (x < (edge[1][0] - edge[0][0]) * (y - edge[0][1]) / (edge[1][1] - edge[0][1]) + edge[0][0]):
                    intersections += 1
            if intersections % 2 == 1:
                return True, idx
        return False, 0
    
    def is_outside_boundary(self, s1):
        x1_min, y1_min = min(s1, key=lambda p: p[0])[0], min(s1, key=lambda p: p[1])[1]
        x1_max, y1_max = max(s1, key=lambda p: p[0])[0], max(s1, key=lambda p: p[1])[1]
        x2_min, y2_min = self.min_coord, self.min_coord 
        x2_max, y2_max = self.max_coord, self.max_coord 
        
        # Check if any part of the first square is outside world boundary
        if x1_min < x2_min or x1_max > x2_max or y1_min < y2_min or y1_max > y2_max:
            return True
        return False
    
    def is_pts_identical(self, pt1, pt2, eps=0.0001):
        if abs(pt2[0] - pt1[0]) < eps and abs(pt2[1] - pt1[1]) < eps:
            return True
        return False
    
    def is_pt_in_list(self, pt, pts, eps=0.0001):
        for point in pts:
            if abs(pt[0] - point[0]) < eps and abs(pt[1] - point[1]) < eps:
                return True
        return False

    def generate_terrain(self, width, height, scale, octaves, persistence, lacunarity, seed=0, central_flatness=0.5):
        random.seed(seed)  # Seed for reproducibility
        dx = round(random.uniform(0.1, 1.0), 1)
        dy = round(random.uniform(0.1, 1.0), 1)
        
        terrain = np.zeros((height, width))
        
        center_x = 50
        center_y = 40
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        for y in range(height):
            for x in range(width):
                nx = x / width - dx
                ny = y / height - dy
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max_distance
                flatness = central_flatness * distance
                value = 1
                amplitude = 1
                frequency = 1
                
                for _ in range(octaves):
                    value += amplitude * noise.snoise2(nx * frequency, ny * frequency)
                    amplitude *= persistence
                    frequency *= lacunarity
                    
                terrain[y][x] = value * flatness
        
        terrain = scale * (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))
        
        return terrain

    def plt2arr(self, fig):
        fig.canvas.draw()
        # grab the pixel buffer and dump it into a numpy array
        rgb_array = np.array(fig.canvas.renderer._renderer)
        return rgb_array
    
    def make_screen(self, show_graph=False):
        # Create a figure and axis object
        fig, ax2d = plt.subplots()

        # add eps, step, rwd text to frame
        total_return = float("{0:.3f}".format(self.cumul_rwd))
        fig.suptitle(f"Episode {self.eps} Step {self.stp} Return {total_return}", fontsize=14, 
                     ha='center', va='top', fontweight='normal')

        # add terrain slope gradient 
        topo = ax2d.imshow(self.slopes, cmap='viridis', alpha=0.8)

        # Create a grid of x and y coordinates
        x = np.linspace(0, self.width, self.width)
        y = np.linspace(0, self.height, self.height)
        X, Y = np.meshgrid(x, y)

        # Add contour lines
        ax2d.contour(X, Y, self.terrain_elevations, levels=self.scale, colors='k', alpha=0.6)

        # Create site boundary patch 
        ax2d.add_patch(Polygon(np.array(self.world_boundary), facecolor=(1,1,1,0), edgecolor="k"))

        # Create cells patches and add them to the axis
        for s in self.street_geometries:
            poly_patch = Polygon(np.array(s), facecolor='white', edgecolor='gray', alpha=0.3)
            ax2d.add_patch(poly_patch)

        for h in self.house_geometries:
            poly_patch = Polygon(np.array(h), facecolor='black', edgecolor='gray')
            ax2d.add_patch(poly_patch)
        
        if show_graph:
            # draw graph edges on frame
            nx.draw(self.graph, nx.get_node_attributes(self.graph, 'pos'), with_labels=False, node_size=0, width=2.0, edge_color='w', ax=ax2d)

        ax2d.autoscale()
        ax2d.set_aspect("equal")
        # plt.imshow(frame)

        # Create a 3D plot
        # ax3d = fig.add_subplot(212, projection='3d')
        # ax3d.set_box_aspect([self.width, self.height, self.scale])

        # topo = ax3d.plot_surface(X, Y, self.terrain_elevations, facecolors=plt.cm.cividis(self.slopes), cmap='cividis', alpha=1)

        # Adjust spacing between subplots
        # fig.subplots_adjust(hspace=0.01)
        # Set the width of the subplots to be equal
        # ax2d.set_position([0.1, 0.55, 0.8, 0.4])  # [left, bottom, width, height]
        # ax3d.set_position([0.05, 0.07, 0.9, 0.35])  # [left, bottom, width, height]

        # Add a colorbar to show the slope values
        # fig.colorbar(topo, shrink=0.5, aspect=10)

        # fig.subplots_adjust(top=0.9, bottom=0.05, left=0.01, right=0.99)

        plt.axis('off')

        # np array from plot
        frame = self.plt2arr(fig)

        plt.close()
        return frame

    def close(self): 
        self.isopen = False

# Path Node version 01 env
class GPN_v1(gym.Env):

    """ 
    An OpenAI Gym environment based on the dynamics of the 'Path Node' model (Erickson & Lloyd-Jones, 1997) with 
    NetworkX graph representations of the street network.

    """ 

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    def __init__(self, screen_size=500):
        super(GPN_v1, self).__init__()

        self.screen_width = screen_size
        self.screen_height = screen_size

        self._carrier_color = 0.5
        self._house_color = 0
        self._street_color = 1
        self._cell_size = 3
        self.obs_size = 100

        # define the world boundary 
        self.min_coord, self.max_coord = -50, 50  
        self.world_boundary = [
            [self.min_coord, self.min_coord],
            [self.max_coord, self.min_coord],
            [self.max_coord, self.max_coord],
            [self.min_coord, self.max_coord]
        ]

        # initial cell geometry
        self.centroids = []
        self.cell_geometries = [] 
        self.tobe_cell_geometries = []
        self.house_geometries = []
        self.street_geometries = []

        # initial location
        x, y = 0.0, 0.0
        
        self.cell = [x, y]
        self.adjacent_cells = [[x, y]]
        self.adj_cells = [[x, y]]

        # initial state
        self.cell_states = [] 

        # initial street graph
        self.graph = nx.Graph() 

        self.encode_coord = interpolate.interp1d(
            [self.min_coord - self._cell_size, self.max_coord + self._cell_size],
            [0, 1]
        )

        move_space = [
            [0, 1],
            [-10, -5, 0, 5, 10]
        ]
        self.possible_moves = list(itertools.product(*move_space))

        # define action and observation spacesindex 
        self.action_space = gym.spaces.Discrete(len(self.possible_moves))
        self.observation_space = gym.spaces.Box(low=0, high=1, 
                                                shape=(3*self.obs_size,), 
                                                dtype=np.float32) 

        self.stp = 0
        self.eps = 0
        self.cumul_rwd = 0.0
        self.cumul_perf = 0.0
        self.settlement_size = 0.0

        self.returns = []
        self.performances = []
        self.eps_lengths = []

        self.isopen = True

    def reset(self):
        # initial state
        self.cell_states = []
        
        # reset counters
        self.eps += 1
        self.stp = 0
        self.cumul_rwd = 0.0
        self.cumul_perf = 0.0
        
        # initial location
        x, y = 0.0, 0.0

        self.cell = [x, y]
        self.adjacent_cells = [[x, y]]
        self.adj_cells = [[x, y]]

        # initial cell geometry
        self.centroids = []
        self.cell_geometries = [] 
        self.cell_geometry = self.make_cell_geometry(x, y, self._cell_size)
        self.tobe_cell_geometries = [self.cell_geometry.copy()]
        self.house_geometries = []
        self.street_geometries = []

        # initial street graph
        self.graph = nx.Graph()

        # initial observation
        observation = self.get_obs()
        self.settlement_size = 0.0

        self.returns.append(self.cumul_rwd)
        self.performances.append(self.cumul_perf)
        self.eps_lengths.append(self.stp + 1)
        
        return observation
    
    def step(self, action):
        self.stp += 1

        move = self.possible_moves[action]
        _cell_state = move[0]
        orientation = move[1]

        # rotate current cell geometry
        current_cell_geometry = self.cell_geometry.copy()
        if orientation != 0:
            current_cell_geometry = self.rotate_cell(current_cell_geometry, orientation)

        # update state step
        self.cell_states.append(_cell_state)
        self.centroids.append(self.cell)
        self.cell_geometries.append(current_cell_geometry)

        if _cell_state == 0:
            self.house_geometries.append(current_cell_geometry)
        else:
            self.street_geometries.append(current_cell_geometry)
        
        original_length = len(self.adj_cells)
        del self.adj_cells[0]
        del self.tobe_cell_geometries[0]
        deleted = len(self.adj_cells) - original_length
        if deleted == 0:
            raise RuntimeError("The current action cell must be deleted at each step.")
        
        if _cell_state == 1:
            # get von Neumann neighborhood
            adjacent, adj_geom = self.get_adjacent(current_cell_geometry)

            # Add street node to graph
            node = (self.cell[0], self.cell[1])
            self.graph.add_node(node, pos=node) 

            for i, item in enumerate(adjacent):
                inside_tobes, _ = self.is_pt_in_geomList(item, self.tobe_cell_geometries)
                if inside_tobes:
                    continue
                else:
                    inside, cell_id = self.is_pt_in_geomList(item, self.cell_geometries)
                    if inside:
                        cell_type = self.cell_states[cell_id]
                        if cell_type == 0:
                            continue
                        else:
                            _adj_point = self.centroids[cell_id]
                            adj_node = (_adj_point[0], _adj_point[1])
                            self.graph.add_node(adj_node, pos=adj_node)
                            self.graph.add_edge(node, adj_node) 
                    else:
                        self.adjacent_cells.append(item)
                        self.adj_cells.append(item)
                        self.tobe_cell_geometries.append(adj_geom[i])

        rwd = 0.0
        # performance
        if _cell_state == 0: # house cell
            self.settlement_size += 1.0
            rwd += 1.0

        self.cumul_rwd += rwd  # update episode cumulative reward

        # done
        done = len(self.adj_cells) == 0 or self.is_outside_boundary(self.cell_geometry)

        if len(self.adj_cells) > 0:
            # next action location selection
            self.cell = self.adj_cells[0]  # random.choice(self.adj_cells)
            self.cell_geometry = self.tobe_cell_geometries[0]
        
        # observation
        observation = self.get_obs()

        # log returns and episode lengths
        if done:
            # street network performance
            cycle_perf = 0.0 
            try:
                cycles = nx.minimum_cycle_basis(self.graph)
                n_cycles = len(cycles) 
                # continue only if an additional cycle is detected 
                if n_cycles > 0: 
                    # get lengths of all cycles 
                    cycle_lengths = [len(x) for x in cycles]
                    for i, length in enumerate(cycle_lengths):
                        if length >= 12:
                            cycle_perf += 10.0
            except nx.exception.NetworkXNoCycle as e:
                pass
            
            perf = self.settlement_size + cycle_perf
            self.cumul_perf = perf

            rwd += cycle_perf
            self.cumul_rwd += cycle_perf  # update episode cumulative reward

            self.returns[-1] = self.cumul_rwd
            self.performances[-1] = self.cumul_perf
            self.eps_lengths[-1] = self.stp
        
        return observation, rwd, done, {}

    def render(self, mode="rgb_array"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either 'human' or 'rgb_array'"

        frame = self.make_screen(show_graph=True)

        if mode == "human":
            # cv2.imshow('BeadyRing_v1', frame)
            # cv2.waitKey(0)
            pass

        return frame if mode == "rgb_array" else self.isopen

    def get_obs(self):
        # observation
        obs_ = []
        coord_fill = []

        # get cell states
        obs_ += self.cell_states.copy()[-self.obs_size:]
        if len(obs_) < self.obs_size:
            coord_fill = [0]*(self.obs_size - len(obs_))
            obs_ += [0.5]*(self.obs_size - len(obs_))
            
        # get cell coordinates
        xs = [pt[0] for pt in self.centroids]
        ys = [pt[1] for pt in self.centroids]
        # remap coordinates to [0, 1] range
        xs = self.encode_coord(xs).tolist() + coord_fill
        ys = self.encode_coord(ys).tolist() + coord_fill

        observation = list(itertools.chain(*zip(obs_, xs, ys)))
        observation = np.array(observation, dtype=np.float32).reshape(3*self.obs_size,)
        return observation
    
    def get_adjacent(self, square):
        # von Neumann neighborhood
        side_length = self._cell_size

        # Calculate the direction vector of one of the sides of the square
        dx = square[1][0] - square[0][0]
        dy = square[1][1] - square[0][1]
        side_vector = [dx / side_length, dy / side_length]

        # Calculate the normal vector of the side vector
        normal_vector = [-side_vector[1], side_vector[0]]

        # Calculate the coordinates of the adjacent squares
        top_square = [[square[0][0] - normal_vector[0] * side_length, square[0][1] - normal_vector[1] * side_length],
                    [square[1][0] - normal_vector[0] * side_length, square[1][1] - normal_vector[1] * side_length],
                    [square[2][0] - normal_vector[0] * side_length, square[2][1] - normal_vector[1] * side_length],
                    [square[3][0] - normal_vector[0] * side_length, square[3][1] - normal_vector[1] * side_length]]
        top_center = self.center_of_square(top_square)

        right_square = [[square[1][0] + side_vector[0] * side_length, square[1][1] + side_vector[1] * side_length],
                        [square[2][0] + side_vector[0] * side_length, square[2][1] + side_vector[1] * side_length],
                        [square[3][0] + side_vector[0] * side_length, square[3][1] + side_vector[1] * side_length],
                        [square[0][0] + side_vector[0] * side_length, square[0][1] + side_vector[1] * side_length]]
        right_center = self.center_of_square(right_square)

        bottom_square = [[square[0][0] + normal_vector[0] * side_length, square[0][1] + normal_vector[1] * side_length],
                        [square[1][0] + normal_vector[0] * side_length, square[1][1] + normal_vector[1] * side_length],
                        [square[2][0] + normal_vector[0] * side_length, square[2][1] + normal_vector[1] * side_length],
                        [square[3][0] + normal_vector[0] * side_length, square[3][1] + normal_vector[1] * side_length]]
        bottom_center = self.center_of_square(bottom_square)

        left_square = [[square[3][0] - side_vector[0] * side_length, square[3][1] - side_vector[1] * side_length],
                    [square[0][0] - side_vector[0] * side_length, square[0][1] - side_vector[1] * side_length],
                    [square[1][0] - side_vector[0] * side_length, square[1][1] - side_vector[1] * side_length],
                    [square[2][0] - side_vector[0] * side_length, square[2][1] - side_vector[1] * side_length]]
        left_center = self.center_of_square(left_square)
        
        return [top_center, right_center, bottom_center, left_center], [top_square, right_square, bottom_square, left_square]
    
    def make_cell_geometry(self, x, y, side_len):
        # Calculate half the length of the square's side
        half_side_len = side_len / 2
        
        # Calculate the x and y coordinates of each corner relative to the center point
        top_left = [x - half_side_len, y + half_side_len]
        top_right = [x + half_side_len, y + half_side_len]
        bottom_left = [x - half_side_len, y - half_side_len]
        bottom_right = [x + half_side_len, y - half_side_len]
        
        # Return the coordinates of each corner as a nested list
        return [bottom_left, bottom_right, top_right, top_left]
    
    def rotate_cell(self, coords, angle):
        # Calculate the center of the square
        x_center = self.cell[0]
        y_center = self.cell[1]

        # Convert the angle to radians
        theta = radians(angle)

        # Rotate each corner of the square around the center
        rotated_coords = []
        for coord in coords:
            x = coord[0] - x_center
            y = coord[1] - y_center
            x_new = x * cos(theta) - y * sin(theta)
            y_new = x * sin(theta) + y * cos(theta)
            rotated_coords.append([x_new + x_center, y_new + y_center])

        return rotated_coords
    
    def center_of_square(self, coords):
        # Calculate the center point of the square using the average of the x and y coordinates of the corners
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        center_x = (max(x_coords) + min(x_coords)) / 2
        center_y = (max(y_coords) + min(y_coords)) / 2
        # Return the coordinates of the center point as a tuple
        return [center_x, center_y]
    
    def is_pt_in_geomList(self, pt, squares):
        x, y = pt[0], pt[1]

        for idx, square_corners in enumerate(squares):
            # Define the edges of the square
            edges = []
            for i in range(len(square_corners)):
                if i == len(square_corners)-1:
                    edges.append([square_corners[i], square_corners[0]])
                else:
                    edges.append([square_corners[i], square_corners[i+1]])

            # Check if the point is inside the square
            intersections = 0
            for edge in edges:
                if ((edge[0][1] > y) != (edge[1][1] > y)) and \
                (x < (edge[1][0] - edge[0][0]) * (y - edge[0][1]) / (edge[1][1] - edge[0][1]) + edge[0][0]):
                    intersections += 1
            if intersections % 2 == 1:
                return True, idx
        return False, 0
    
    def is_outside_boundary(self, s1):
        x1_min, y1_min = min(s1, key=lambda p: p[0])[0], min(s1, key=lambda p: p[1])[1]
        x1_max, y1_max = max(s1, key=lambda p: p[0])[0], max(s1, key=lambda p: p[1])[1]
        x2_min, y2_min = self.min_coord, self.min_coord 
        x2_max, y2_max = self.max_coord, self.max_coord 
        
        # Check if any part of the first square is outside world boundary
        if x1_min < x2_min or x1_max > x2_max or y1_min < y2_min or y1_max > y2_max:
            return True
        return False
    
    def is_pts_identical(self, pt1, pt2, eps=0.0001):
        if abs(pt2[0] - pt1[0]) < eps and abs(pt2[1] - pt1[1]) < eps:
            return True
        return False
    
    def is_pt_in_list(self, pt, pts, eps=0.0001):
        for point in pts:
            if abs(pt[0] - point[0]) < eps and abs(pt[1] - point[1]) < eps:
                return True
        return False

    def plt2arr(self, fig):
        fig.canvas.draw()
        # grab the pixel buffer and dump it into a numpy array
        rgb_array = np.array(fig.canvas.renderer._renderer)
        return rgb_array
    
    def make_screen(self, show_graph=False):
        # Create a figure and axis object
        fig, ax = plt.subplots()

        # Create site boundary patch 
        ax.add_patch(Polygon(np.array(self.world_boundary), facecolor="white", edgecolor="gray"))

        # Create cells patches and add them to the axis
        for s in self.street_geometries:
            poly_patch = Polygon(np.array(s), facecolor='white', edgecolor='gray')
            ax.add_patch(poly_patch)

        for h in self.house_geometries:
            poly_patch = Polygon(np.array(h), facecolor='black', edgecolor='gray')
            ax.add_patch(poly_patch)
        
        if show_graph:
            # draw graph edges on frame
            nx.draw(self.graph, nx.get_node_attributes(self.graph, 'pos'), with_labels=False, node_size=0, width=1.0, edge_color='b', ax=ax)

        # plt.imshow(frame)

        ax.autoscale()
        ax.set_aspect("equal")

        # add eps, step, rwd text to frame
        total_return = float("{0:.3f}".format(self.cumul_rwd))
        plt.title(f"Episode {self.eps} Step {self.stp} Return {total_return}", fontsize=14, loc='center', fontweight='normal')

        # np array from plot
        frame = self.plt2arr(fig)

        plt.close()
        return frame

    def close(self): 
        self.isopen = False


