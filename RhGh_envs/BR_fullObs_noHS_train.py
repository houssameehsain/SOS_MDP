import socket
import struct
import pickle
import numpy as np
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
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 10}

    def __init__(self):
        super(Env, self).__init__()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        addr = ("127.0.0.1", 50710)
        s.bind(addr)
        s.listen(1)
        clientsocket, _ = s.accept()

        self._socket = clientsocket
        self._conn = Connection(clientsocket)

        self.grid_len = 36 # Make sure grid size matches max_row_len in the Gh env
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                shape=(1, self.grid_len, self.grid_len), 
                                                dtype=np.uint8) 

    def reset(self):
        self._conn.send_object("reset")
        msg = self._conn.receive_object()

        return np.asarray(msg["state"]).reshape(1, self.grid_len, self.grid_len)

    def step(self, action):
        self._conn.send_object(action.item())
        msg = self._conn.receive_object()
        obs = np.asarray(msg["state"]).reshape(1, self.grid_len, self.grid_len)
        rwd = msg["reward"]
        done = msg["done"]
        info = msg["info"]
        return obs, rwd, done, info

    def render(self, mode='rgb_array'):
        msg = self._conn.receive_object()
        img = np.asarray(msg["state"]).reshape(self.grid_len, self.grid_len, 1)
        if mode == 'human':
            plt.imshow(img)
        elif mode == 'rgb_array':
            return img
        else:
            raise RuntimeError("Invalid render mode")

    def close(self):
        self._conn.send_object("close")
        self._socket.close()


if __name__ == '__main__':
    # Log in to W&B account
    print('Wandb login ...')
    wandb.login(key='') # place wandb key here!

    config = {
        "rl_alg": "DQN",
        "policy_type": "CnnPolicy",
        "total_timesteps": 10000000
    }

    run = wandb.init(
        entity='', #Replace with your wandb entity & project
        project="BeadyRing_DRL",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,
        name=f'{config["rl_alg"]}_{config["policy_type"]}'
    )

    print("\n   Set 'Loop' boolean toggle in Gh to True ... \n")

    def make_env():
        env = Env()
        # check_env(env) # check if the env follows the gym interface
        env = Monitor(env)  # record stats
        return env

    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(env, f"videos/{run.id}", 
                           record_video_trigger=lambda x: x % ((36**2)*10) == 0, 
                           video_length=36**2, name_prefix='BR_RL-train') # replace grid_len to match env

    model = DQN(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}", device='cuda')
    model.learn(total_timesteps=config["total_timesteps"],
                callback=WandbCallback(
                    model_save_path=f'models/{run.id}', 
                    model_save_freq=100,
                    verbose=2
                    )    
                )

    cum_rwd = 0
    obs = env.reset()
    for i in range(300):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cum_rwd += reward
        if done:
            obs = env.reset()
            print("Return = ", cum_rwd)
            cum_rwd = 0
    env.close()

    run.finish()



