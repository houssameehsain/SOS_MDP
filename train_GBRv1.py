import os
import numpy as np
import cv2 
import json 
from toolz.itertoolz import interleave

import torch.nn as nn
from stable_baselines3 import DQN, PPO

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback, CallbackList 
from stable_baselines3.common.env_checker import check_env   

from envs import GBR_v1


run_id = "gbrv1_dqn_moore_01" 
evaluate = False 

config = {
    "rl_alg": "DQN",
    "policy_type": "MlpPolicy",
    "n_workers": 32,  # number of parallel processes to use 
    "total_timesteps": 3000000,  # total number of steps
    "train_log_eps_freq": 1, 
    "train_render_eps_freq": 50,
    "n_eval_eps": 100,
    "train_run_dir": f"./logs/run_{run_id}/train/",
    "eval_run_dir": f"./logs/run_{run_id}/eval/",
    "plot_dir": f"./logs/run_{run_id}/train/plots/",
    "model_dir": f"./logs/run_{run_id}/models/"
}

class TrainCallback(BaseCallback):
    def __init__(self, model, n_envs, 
                 log_freq: int, render_freq: int, 
                 render_dir: str, model_log_dir: str, 
                 verbose: int = 0):
        super(TrainCallback, self).__init__(verbose)
        self.n_envs = n_envs
        self.model = model
        self.envs = model.get_env()

        self.train_log_eps_freq = log_freq
        self.render_freq = render_freq
        self.dir = render_dir

        self.best_mean_reward = -np.inf
        self.save_path = os.path.join(model_log_dir, "best_model")

    def _on_step(self) -> bool:
        eps = self.envs.get_attr('eps', indices=[-1])[-1]
        stp = self.envs.get_attr('stp', indices=[-1])[-1]

        if eps % self.train_log_eps_freq == 0: 
            if stp == 1: 
                returns = list(interleave(self.envs.get_attr('returns', indices=[i for i in range(self.n_envs)])))
                if len(returns) > 0: 
                    scores = list(interleave(self.envs.get_attr('performances', indices=[i for i in range(self.n_envs)])))
                    eps_lengths = list(interleave(self.envs.get_attr('eps_lengths', indices=[i for i in range(self.n_envs)])))

                    # log rewards and scores
                    perfLog = {
                        "scores": scores,
                        "returns": returns,
                        "eps_lengths": eps_lengths
                    } 
                    json_obj = json.dumps(perfLog, indent=4)
                    with open(f"{self.dir}plots/performance_log.json", "w") as outfile:
                        outfile.write(json_obj)

                    # Mean training reward over the last 100 episodes
                    mean_reward = np.mean(returns[-100:])
                    # New best model, you could save the agent here
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        # save best model
                        if self.verbose >= 1:
                            print(f"Eps {eps} Best Avg Rwd {self.best_mean_reward:.3f} | Saving new best model to {self.save_path}")
                        self.model.save(self.save_path)

        if eps % self.render_freq == 0: 
            # log renders
            frame = self.envs.get_images()[-1]
            bgr_array = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(f"{self.dir}{eps}-{stp}.png", bgr_array)

        return True  


if __name__ == '__main__': 
    os.makedirs(config["plot_dir"], exist_ok=True)
    os.makedirs(config["model_dir"], exist_ok=True)
    os.makedirs(config["eval_run_dir"], exist_ok=True)
    os.makedirs(config["train_run_dir"], exist_ok=True)

    # multiprocess training 
    env = make_vec_env(lambda: GBR_v1(local=True, screen_size=500),
                        n_envs=config["n_workers"], 
                        vec_env_cls=SubprocVecEnv)

    # env = Monitor(env, config["train_run_dir"])
    # check_env(env) # check if the env follows the gym interface 

    if config["rl_alg"] == 'DQN':
        model = DQN(policy = config["policy_type"],
                    env = env,
                    gamma = 0.99,
                    learning_rate = 0.000102253935048113,
                    batch_size = 256,
                    buffer_size = 50000,
                    train_freq = 128,
                    gradient_steps = 128,
                    exploration_fraction = 0.00691378781870256,
                    exploration_final_eps = 0.0044765306701284,
                    target_update_interval = 20000,
                    learning_starts = 5000,
                    policy_kwargs = dict(net_arch=[64]),
                    device = 'cpu',  # 'cuda' 'cpu'
                    verbose=0)  # verbose=2 for debugging
        
    elif config["rl_alg"] == 'PPO':
        model = PPO(policy = config["policy_type"],
                    env = env,
                    learning_rate = 0.00010440567930314779, 
                    n_steps = 256, 
                    batch_size = 64, 
                    n_epochs = 20, 
                    gamma = 0.999, 
                    gae_lambda = 0.92, 
                    clip_range = 0.4, 
                    ent_coef = 2.9945927354731036e-07, 
                    vf_coef = 0.0009499078688032755, 
                    max_grad_norm = 0.3, 
                    policy_kwargs = dict(
                        net_arch=dict(pi=[256, 256], vf=[256, 256]),
                        activation_fn=nn.Tanh,
                        ortho_init=True,
                    ),
                    device = 'cpu',  # 'cuda' 'cpu'
                    verbose=0)  # verbose=2 for debugging
        

    if not evaluate:
        # callbacks
        trainLog = TrainCallback(model, 
                                config["n_workers"], 
                                config["train_log_eps_freq"],
                                config["train_render_eps_freq"], 
                                config["train_run_dir"],
                                config['model_dir'], 
                                verbose=1)
        progressBar = ProgressBarCallback()

        # Training 
        model.learn(
            total_timesteps=config["total_timesteps"],  # // config["n_workers"]
            callback=CallbackList([
                trainLog, 
                progressBar
            ])
        ) 

    env.close()

    # Evaluate the trained agent
    if evaluate:
        trained_eval_env = Monitor(GBR_v1(local=True, screen_size=500))

        # del and reload trained model 
        del model
        model = PPO.load(f"{config['model_dir']}best_model", env=trained_eval_env, print_system_info=False)

        mean_reward, std_reward = evaluate_policy(model, trained_eval_env, n_eval_episodes=config["n_eval_eps"])
        print(f'Trained agent | Mean reward: {mean_reward} +/- {std_reward:.2f}')

        trained_eval_env.close() 