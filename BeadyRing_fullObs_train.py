import socket
import struct
import pickle

import numpy as np
import cv2
import pygame
from pygame import gfxdraw
from PIL import Image
import matplotlib.pyplot as plt

import gym
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.env_checker import check_env
import wandb
from wandb.integration.sb3 import WandbCallback


class Connection:
    def __init__(self, s):
        self._socket = s
        self._buffer = bytearray()

    def receive_object(self):
        while len(self._buffer) < 4 or len(self._buffer) < struct.unpack("<L", self._buffer[:4])[0] + 4:
            new_bytes = self._socket.recv(16)
            if len(new_bytes) == 0:
                return None
            self._buffer += new_bytes
        length = struct.unpack("<L", self._buffer[:4])[0]
        _, body = self._buffer[:4], self._buffer[4:length + 4]
        obj = pickle.loads(body)
        self._buffer = self._buffer[length + 4:]
        return obj

    def send_object(self, d):
        body = pickle.dumps(d, protocol=2)
        header = struct.pack("<L", len(body))
        msg = header + body
        self._socket.send(msg)

class Env(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self):
        super(Env, self).__init__()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        addr = ("127.0.0.1", 50710)
        s.bind(addr)
        s.listen(1)
        clientsocket, _ = s.accept()

        self._socket = clientsocket
        self._conn = Connection(clientsocket)

        self.cell_size = 10
        self.grid_len = 36 # Make sure grid size matches max_row_len in the Gh env
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                shape=(1, self.grid_len, self.grid_len), 
                                                dtype=np.uint8) 

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

    def reset(self):
        self._conn.send_object("reset")
        msg = self._conn.receive_object()
        self.state = np.asarray(msg["state"]).reshape(1, self.grid_len, self.grid_len)
        return self.state

    def step(self, action):
        self._conn.send_object(action.item())
        msg = self._conn.receive_object()
        self.state = np.asarray(msg["state"]).reshape(1, self.grid_len, self.grid_len)
        rwd = msg["reward"]
        done = msg["done"]
        info = msg["info"]
        return self.state, rwd, done, info

    def render(self, mode='human'):
        screen_width = self.grid_len * self.cell_size
        screen_height = self.grid_len * self.cell_size

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption('BeadyRing DRL train')

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state is None:
            return None

        img = self.state.reshape(self.grid_len, self.grid_len)
        img = cv2.resize(img, (screen_width, screen_height), interpolation = cv2.INTER_AREA)
        frame = np.zeros([screen_width, screen_height, 3], dtype=np.uint8)
        frame[:,:,0] = img.astype(np.uint8)
        frame[:,:,1] = img.astype(np.uint8)
        frame[:,:,2] = img.astype(np.uint8)

        self.surf = pygame.surfarray.make_surface(frame)
        self.screen.blit(self.surf, (0, 0))

        if mode == 'human':
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == 'rgb_array':
            return frame # np.array(pygame.surfarray.pixels3d(self.screen))

        else:
            return self.isopen

    def close(self):
        self._conn.send_object("close")
        self._socket.close()
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


if __name__ == '__main__':
    # Log in to W&B account
    print('Wandb login ...')
    wandb.login(key='a62c193ea97080a59a7f646248cd9ec23346c61c') # place wandb key here!

    config = {
        "rl_alg": "DQN",
        "policy_type": "CnnPolicy",
        "total_timesteps": 10000000
    }

    run = wandb.init(
        entity='hehsain', #Replace with your wandb entity & project
        project="BeadyRing_DRL",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,
        name=f'{config["rl_alg"]}_{config["policy_type"]}'
    )

    print('\n   Reset and Loop HoopSnake Gh component ... \n')

    def make_env():
        env = Env()
        # check_env(env) # check if the env follows the gym interface
        env.reset() # reset
        env = Monitor(env)  # record stats
        return env

    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(env, f"videos/{run.id}", 
                           record_video_trigger=lambda x: x % 12960 == 0, 
                           video_length=1296, name_prefix='BR_RL-train') # replace grid_len to match env

    model = DQN(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}", device='cuda')
    model.learn(total_timesteps=config["total_timesteps"],
                callback=WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=f'models/{run.id}', 
                    model_save_freq=100,
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



