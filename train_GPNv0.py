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

from envs import GPN_v0


run_id = "gpnv0_ppo_10_04" 
evaluate = False 

config = {
    "rl_alg": "PPO",
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
                    mean_reward = np.mean(scores[-100:])
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
    env = make_vec_env(lambda: GPN_v0(screen_size=500),
                        n_envs=config["n_workers"], 
                        vec_env_cls=SubprocVecEnv)

    # env = Monitor(env, config["train_run_dir"])
    # check_env(env) # check if the env follows the gym interface 

    if config["rl_alg"] == 'DQN':
        model = DQN(policy = config["policy_type"],
                    env = env,
                    gamma = 0.99,
                    learning_rate = 0.039950128395957,
                    batch_size = 64,
                    buffer_size = 100000,
                    train_freq = 128,
                    gradient_steps = 16,
                    exploration_fraction = 0.00336290481089121,
                    exploration_final_eps = 0.0123340810433939,
                    target_update_interval = 15000,
                    learning_starts = 10000,
                    policy_kwargs = dict(net_arch=[64, 64]),
                    device = 'cpu',  # 'cuda' 'cpu'
                    verbose=0)  # verbose=2 for debugging

    elif config["rl_alg"] == 'PPO':
        model = PPO(policy = config["policy_type"],
                    env = env,
                    learning_rate = 0.000121136184813815, 
                    n_steps = 512, 
                    batch_size = 64, 
                    n_epochs = 10, 
                    gamma = 0.995, 
                    gae_lambda = 0.98, 
                    clip_range = 0.4, 
                    ent_coef = 4.03008700396656E-08, 
                    vf_coef = 0.000496796993945126, 
                    max_grad_norm = 0.9, 
                    policy_kwargs = dict(
                        net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]),
                        activation_fn=nn.Tanh,
                        ortho_init=False,
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
        trained_eval_env = Monitor(GPN_v0(screen_size=500))

        # del and reload trained model 
        del model
        model = PPO.load(f"{config['model_dir']}best_model", env=trained_eval_env, print_system_info=False)

        mean_reward, std_reward = evaluate_policy(model, trained_eval_env, n_eval_episodes=config["n_eval_eps"])
        print(f'Trained agent | Mean reward: {mean_reward} +/- {std_reward:.2f}')

        trained_eval_env.close() 
