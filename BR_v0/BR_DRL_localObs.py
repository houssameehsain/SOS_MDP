import os
import numpy as np
from math import floor
import cv2

import gym
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.env_checker import check_env
import wandb
from wandb.integration.sb3 import WandbCallback


class BeadyRing(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    def __init__(self, run):
        super(BeadyRing, self).__init__()

        self._carrier_color = 127.5
        self._house_color = 0
        self._street_color = 255
        self._cell_size = 3
        self._max_row_len = 36
        self._obs_size = 3
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
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                shape=(1, self._obs_size**2), 
                                                dtype=np.uint8) 
        
        self.run = run
        self.path = f"D:/RLinGUD/BeadyRing/images/{self.run.id}/"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        self.stp = 0
        self.eps = 0
        self.cumul_rwd = 0

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

        # adj_street_count = 0
        # for a in adjacent:
        #     if int(self.state[a[0] + self.pad, a[1] + self.pad]) == 255:
        #         adj_street_count += 1

        if _cell_state == 0: # house cell
            reward += 1/(self._max_row_len**2) # density 
        self.cumul_rwd += reward

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
        
        # save to project dir
        if self.eps == 1 or self.eps % 1000 == 0:
            frame = self.make_screen()
            cv2.imwrite(self.path + f"eps-{self.eps}_step-{self.stp}.png", frame)

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

    def render(self, mode="human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either 'human' or 'rgb_array'"

        frame = self.make_screen()

        if mode == "human":
            cv2.imshow('Beady Ring', frame)
            cv2.waitKey(0)

        return frame if mode == "rgb_array" else self.isopen

    def get_obs(self):
        # observation
        left_up_R = int(self.R - self.pad)
        left_up_C = int(self.C - self.pad)
        right_bottom_R = int(self.R + self.pad)
        right_bottom_C = int(self.C + self.pad)
        obs_ = self.state[left_up_R:right_bottom_R + 1, left_up_C:right_bottom_C + 1]
        return obs_.reshape(1, self._obs_size**2)

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
        screen_width = 500
        screen_height = 500

        img = self.state[self.pad:self._max_row_len+self.pad, self.pad:self._max_row_len+self.pad]
        img = img.reshape(self._max_row_len, self._max_row_len)
        img = cv2.resize(img, (screen_width, screen_height), interpolation = cv2.INTER_AREA)
        frame = np.zeros([screen_width, screen_height, 3], dtype=np.uint8)
        frame[:,:,0] = img.astype(np.uint8)
        frame[:,:,1] = img.astype(np.uint8)
        frame[:,:,2] = img.astype(np.uint8)

        frame = cv2.copyMakeBorder(frame, 30, 5, 5, 5, cv2.BORDER_CONSTANT, None, value = [127, 127, 127])
        total_return = float("{0:.3f}".format(self.cumul_rwd))
        frame = cv2.putText(frame, f"episode {self.eps} | step {self.stp} | return {total_return}", 
                        (5,20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255, 255), 1)
        return frame

    def close(self):
        self.isopen = False


if __name__ == '__main__':
    # Log in to W&B account
    print('Wandb login ...')
    wandb.login(key='a62c193ea97080a59a7f646248cd9ec23346c61c') # place wandb key here!

    config = {
        "rl_alg": "A2C",
        "policy_type": "MlpPolicy",
        "total_timesteps": 10000000
    }

    run = wandb.init(
        entity='hehsain', #Replace with your wandb entity & project
        project="BeadyRing_DRL",
        config=config,
        sync_tensorboard=True, # auto-upload sb3's tensorboard metrics
        monitor_gym=True,
        name=f'{config["rl_alg"]}_{config["policy_type"]}'
    )

    def make_env():
        env = BeadyRing(run)
        # check_env(env) # check if the env follows the gym interface
        env = Monitor(env)  # record stats
        return env

    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(env, f"videos/{run.id}", 
                           record_video_trigger=lambda x: x % 200000 == 0, 
                           video_length=1500, name_prefix='BR_RL-train') 

    model = A2C(config["policy_type"], env, verbose=1, 
                tensorboard_log=f"runs/{run.id}", device='cuda')
    model.learn(total_timesteps=config["total_timesteps"],
                callback=WandbCallback(
                    gradient_save_freq=500,
                    model_save_path=f'models/{run.id}', 
                    model_save_freq=500,
                    verbose=2
                    )    
                )

    # cum_rwd = 0
    # obs = env.reset()
    # for i in range(300):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     cum_rwd += reward
    #     if done:
    #         obs = env.reset()
    #         print("Return = ", cum_rwd)
    #         cum_rwd = 0
    env.close()

    run.finish()