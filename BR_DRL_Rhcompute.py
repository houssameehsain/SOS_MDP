import os
import numpy as np
from math import floor
import cv2

import rhino3dm
import requests
import base64
import json
import compute_rhino3d.Util

import rhinoinside
rhinoinside.load()
import System
import Rhino 

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
        # establish rh compute connection
        compute_rhino3d.Util.url = 'http://localhost:8081/'
        compute_rhino3d.Util.authToken = ''

        self.post_url = compute_rhino3d.Util.url + 'grasshopper'

        ## Read grasshopper .ghx file ########################### Replace this path
        gh_data = open('D:/RLinGUD/BeadyRing/BR_localObs_Rhcompute_env.ghx', mode='r', encoding='utf-8-sig').read()
        data_bytes = gh_data.encode('utf-8')
        encoded = base64.b64encode(data_bytes)
        self.algo = encoded.decode('utf-8')

        self._obs_size = 3 # make sure these match .ghx file env
        self._max_row_len = 36
        self.pad = int(floor(self._obs_size/2))

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                shape=(1, self._obs_size, self._obs_size), 
                                                dtype=np.uint8) 

        self.res = None
        self.viewer = None
        self.isopen = True

        self.run = run
        self.path = f"D:/RLinGUD/BeadyRing/images/{self.run.id}/"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.eps = 0
        self.stp = 0
        self.cumul_rwd = 0

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        assert self.res is not None, "Call reset before using step method."

        observation, reward, done, info = self.Rhcompute_connect(action=action)

        self.stp += 1
        self.cumul_rwd += reward
        if done:
            self.eps += 1

        return observation, reward, done, info

    def reset(self):
        # initial observation
        observation = self.Rhcompute_connect(reset=True)
        # reset counters
        self.stp = 0
        self.cumul_rwd = 0

        return observation

    def render(self, mode="rgb_array"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either 'human' or 'rgb_array'"

        screen_width = 500
        screen_height = 500

        # get display array
        frame = []
        for output in self.res['values']:
            paramName = output['ParamName']
            InnerTree = output['InnerTree']
            if paramName == 'RH_OUT:state':
                for _, InnerVals in InnerTree.items():
                    row = []
                    for val in InnerVals:
                        data = json.loads(val['data'])
                        row.append(data)
                    frame.append(row)
        frame = np.asarray(frame)
        frame = frame[self.pad:self._max_row_len+self.pad, self.pad:self._max_row_len+self.pad]
        frame = frame.reshape(self._max_row_len, self._max_row_len)
        frame = cv2.resize(frame, (screen_width, screen_height), interpolation = cv2.INTER_AREA)
        screen = np.zeros((screen_width, screen_height, 3), dtype=np.uint8)
        screen[:,:,0] = frame.astype(np.uint8)
        screen[:,:,1] = frame.astype(np.uint8)
        screen[:,:,2] = frame.astype(np.uint8)

        img = cv2.copyMakeBorder(screen, 30, 5, 5, 5, cv2.BORDER_CONSTANT, None, value = [255, 255, 255])
        img = cv2.putText(img, f"eps {self.eps} | step {self.stp} | return {self.cumul_rwd}", 
                        (5,20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0, 255), 1)
        # save to project dir
        cv2.imwrite(self.path + f"eps-{self.eps}_step-{self.stp}.png", img)

        if mode == "human":
            cv2.imshow('Beady Ring', img)
            cv2.waitKey(0)

        return img if mode == "rgb_array" else self.isopen

    def Rhcompute_connect(self, action=None, reset=False):
        if action is not None:
            gh_json = {
                'algo': self.algo,
                'pointer': None,
                'values': [
                    {
                        'ParamName': 'RH_IN:action',
                        'InnerTree': {
                            '{ 0; }': [{'type': 'System.Int32', 'data': str(int(action))}]
                        }
                    }
                ]
            }
        elif reset:
            gh_json = {
                'algo': self.algo,
                'pointer': None,
                'values': [
                    {
                        'ParamName': 'RH_IN:reset',
                        'InnerTree': {
                            '{ 0; }': [{'type': 'System.Boolean', 'data': reset}]
                        }
                    }
                ]
            }
        else:
            raise ValueError('Either action or reset must be provided')

        # Send
        response = requests.post(self.post_url, json=gh_json)

        # Receive
        res = response.content.decode('utf-8')
        self.res = json.loads(res)

        observation = None
        reward = None
        done = False
        info = {}

        for output in self.res['values']:
            paramName = output['ParamName']
            InnerTree = output['InnerTree']
            if paramName == 'RH_OUT:observation':
                observation = []
                for _, InnerVals in InnerTree.items():
                    obs = []
                    for val in InnerVals:
                        data = json.loads(val['data'])
                        obs.append(data)
                    observation.append(obs)
                observation = np.asarray(observation).reshape((1, self._obs_size, self._obs_size))
            elif paramName == 'RH_OUT:reward':
                for _, InnerVals in InnerTree.items():
                    for val in InnerVals:
                        data = json.loads(val['data'])
                        reward = data
            elif paramName == 'RH_OUT:done':
                for _, InnerVals in InnerTree.items():
                    for val in InnerVals:
                        data = json.loads(val['data'])
                        done = data
            elif paramName == 'RH_OUT:info':
                for _, InnerVals in InnerTree.items():
                    for val in InnerVals:
                        data = json.loads(val['data'])
                        info = data

        if observation is None:
            raise ValueError('observation cannot be None')

        if reset:
            return observation
        else:
            return observation, reward, done, info
    
    def close(self):
        self.isopen = False


if __name__ == '__main__':
    # Log in to W&B account
    print('Wandb login ...')
    wandb.login(key='a62c193ea97080a59a7f646248cd9ec23346c61c') # place wandb key here!

    config = {
        "rl_alg": "PPO",
        "policy_type": "MlpPolicy",
        "total_timesteps": 5000000
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
                           record_video_trigger=lambda x: x % 12960 == 0, 
                           video_length=1296, name_prefix='BR_RL-train') # replace grid_len to match env

    model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}", device='cuda')
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

