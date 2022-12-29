from envs import BR_v0, BR_v1

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.env_checker import check_env
import wandb
from wandb.integration.sb3 import WandbCallback


if __name__ == '__main__':
    # Log in to W&B account
    print('Wandb login ...')
    wandb.login(key='a62c193ea97080a59a7f646248cd9ec23346c61c')  # place wandb key here!

    config = {
        "rl_alg": "A2C",
        "policy_type": "MlpPolicy",
        "total_timesteps": 1000,
        "vid_name_prefix": "BR_DRL-train",
        "record_video_freq": 100,
        "video_length": 10,
        "model_save_freq": 500,
    }

    run = wandb.init(
        entity='hehsain',  # place with wandb entity here
        project="sos-mdp",  # place with wandb project name here
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,
        name=f'{config["rl_alg"]}_{config["policy_type"]}'
    )

    def make_env():
        env = BR_v0(run, save_img_freq=100, local=True)  # select a gym env from envs.py
        # check_env(env)  # check if the env follows the gym interface
        env = Monitor(env)  # record stats
        return env

    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(env, f"videos/{run.id}", 
                           record_video_trigger=lambda x: x % config["record_video_freq"] == 0, 
                           video_length=config["video_length"], 
                           name_prefix=config["vid_name_prefix"]) 

    model = A2C(config["policy_type"], env, verbose=1, 
                tensorboard_log=f"runs/{run.id}", device='cuda')
    model.learn(total_timesteps=config["total_timesteps"],
                callback=WandbCallback(
                    gradient_save_freq=config["model_save_freq"],
                    model_save_path=f'models/{run.id}', 
                    model_save_freq=config["model_save_freq"],
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