import os
import numpy as np
import cv2
import socket
import struct
import pickle

import gym
from stable_baselines3 import A2C, PPO
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
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    def __init__(self):
        super(Env, self).__init__()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        addr = ("127.0.0.1", 50710)
        s.bind(addr)
        s.listen(1)
        clientsocket, _ = s.accept()

        self._socket = clientsocket
        self._conn = Connection(clientsocket)

        self.neighborCount = 20 # Make sure obs neighbors count matches self.neighborCount in the Gh env
        self.rotationSpaceLen = 13 # length of the cell rotation action space, verify match with Gh env
        self.action_space = gym.spaces.MultiDiscrete([2, self.rotationSpaceLen])
        self.observation_space = gym.spaces.Box(low=0, high=1, 
                                                shape=(1, 3*self.neighborCount), 
                                                dtype=np.float32) 

        self.screen = None
        self.isopen = True

        self.stp = 0
        self.eps = 0
        self.cumul_rwd = 0

    def reset(self):
        self.eps += 1
        self.stp = 0
        self.cumul_rwd = 0
        self._conn.send_object("reset")
        msg = self._conn.receive_object()
        state = np.asarray(msg["state"]).reshape(1, 3*self.neighborCount)
        return state

    def step(self, action):
        self.stp += 1
        self._conn.send_object(action.tolist())
        msg = self._conn.receive_object()
        state = np.asarray(msg["state"]).reshape(1, 3*self.neighborCount)
        rwd = msg["reward"]
        self.cumul_rwd += rwd
        done = msg["done"]
        info = msg["info"]
        return state, rwd, done, info

    def render(self, mode="human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either 'human' or 'rgb_array'"

        self._conn.send_object("render")
        msg = self._conn.receive_object()
        fpPerspective = msg["visPerspective"]
        fpTop = msg["visTop"]

        frame = self.make_screen(fpPerspective, fpTop)

        if mode == "human":
            cv2.imshow('EJLv0', frame)
            cv2.waitKey(0)

        return frame if mode == "rgb_array" else self.isopen
    
    def make_screen(self, fp1, fp2):
        img1 = cv2.imread(fp1)
        frame1 = self.write_onImage(img1)
        try:
            os.remove(fp1)
        except: pass
        filename1 = f'D:/RLinGUD/EricksonLloyd-Jones/EJLperspective_train/eps_{self.eps}_iter{self.stp}_rwd_{self.cumul_rwd}.png'
        cv2.imwrite(filename1, frame1)

        img2 = cv2.imread(fp2)
        frame2 = self.write_onImage(img2)
        try:
            os.remove(fp2)
        except: pass
        filename2 = f'D:/RLinGUD/EricksonLloyd-Jones/EJLtop_train/eps_{self.eps}_iter{self.stp}_rwd_{self.cumul_rwd}.png'
        cv2.imwrite(filename2, frame2)

        concatImg  = cv2.vconcat([img1, img2])
        concatFrame = self.write_onImage(concatImg)
        concatfp = f'D:/RLinGUD/EricksonLloyd-Jones/EJLconcat_train/eps_{self.eps}_iter{self.stp}_rwd_{self.cumul_rwd}.png'
        cv2.imwrite(concatfp, concatFrame)
        return concatFrame

    def write_onImage(self, img):
        frame = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, None, value = [255, 255, 255])
        total_return = float("{0:.3f}".format(self.cumul_rwd))
        frame = cv2.putText(frame, f"episode {self.eps} | step {self.stp} | return {total_return}", 
                        (20,30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0, 0), 1)
        return frame

    def close(self):
        self._conn.send_object("close")
        self._socket.close()
        if self.screen is not None:
            self.isopen = False


if __name__ == '__main__':
    # Log in to W&B account
    print('Wandb login ...')
    wandb.login(key='a62c193ea97080a59a7f646248cd9ec23346c61c') # place wandb key here!

    config = {
        "rl_alg": "DQN",
        "policy_type": "MlpPolicy",
        "total_timesteps": 10000000
    }

    run = wandb.init(
        entity='hehsain', #Replace with your wandb entity & project
        project="EJL_DRL",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,
        name=f'{config["rl_alg"]}_{config["policy_type"]}'
    )

    print('\n   Reset and Loop HoopSnake Gh component ... \n')

    def make_env():
        env = Env()
        # check_env(env) # check if the env follows the gym interface
        env.reset() 
        env = Monitor(env)  # record stats
        return env

    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(env, f"videos/{run.id}", 
                           record_video_trigger=lambda x: x % 10000 == 0, 
                           video_length=300, name_prefix='EJLv0_RL-train')

    model = A2C(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}", device='cuda')
    model.learn(total_timesteps=config["total_timesteps"],
                callback=WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=f'D:/RLinGUD/EricksonLloyd-Jones/EJLv0Models/{run.id}', 
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



