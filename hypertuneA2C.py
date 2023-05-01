import pickle as pkl 
from typing import Any, Dict 

import os 
# import gym 
# import torch 
import torch.nn as nn 
from stable_baselines3 import A2C 
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike 
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.callbacks import EvalCallback 
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.vec_env import VecMonitor, VecEnv, SubprocVecEnv, VecNormalize 
from gym.wrappers import TimeLimit 
from envs import GBR_v0, GBR_v1 

import optuna 
from optuna.pruners import MedianPruner 
from optuna.samplers import TPESampler 
# from optuna.visualization import plot_optimization_history, plot_param_importances 


N_TRIALS = 1000
N_JOBS = 1
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(1e5)
N_EVAL_EPISODES = 10
N_ENVS = 12
EVAL_FREQ = max(int(N_TIMESTEPS / N_EVALUATIONS) // N_ENVS, 1)

TIMEOUT = int(60 * 2000)  # 480 minutes

run_id = "hypertune_a2c_brv0_01"
directory = f"./logs/run_{run_id}/"
os.makedirs(directory, exist_ok=True)


def sample_a2c_params(trial):
    """
    Sampler for A2C hyperparams.
    :param trial: (optuna.trial)
    :return: (dict)
    """ 

    gamma = trial.suggest_categorical('gamma', [0.9, 0.98, 0.99, 0.995, 0.999, 0.9999]) 
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]) 
    n_steps = trial.suggest_categorical('n_steps', [16, 32, 64, 128, 256, 512, 1024]) 
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)  
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True) 
    ent_coef = trial.suggest_float('ent_coef', 0.000001, 0.2, log=True) 
    vf_coef = trial.suggest_float('vf_coef', 0.1, 1, log=True)
    normalize_advantage = trial.suggest_categorical('normalize_advantage', [True, False]) 
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "large"]) 
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"]) 

    if net_arch == "tiny": 
        net_arch = dict(pi=[128, 128, 128], vf=[128, 128, 128]) 

    elif net_arch == "small": 
        net_arch = dict(pi=[400, 400, 300], vf=[400, 400, 300]) 

    else: 
        net_arch = dict(pi=[512, 512, 512, 128], vf=[512, 512, 512, 128]) 

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn] 

    return { 
        "n_steps": n_steps, 
        "gamma": gamma, 
        "gae_lambda": gae_lambda, 
        "learning_rate": learning_rate, 
        "ent_coef": ent_coef, 
        'vf_coef': vf_coef, 
        "max_grad_norm": max_grad_norm, 
        "normalize_advantage": normalize_advantage, 
        "policy_kwargs": { 
            "net_arch": net_arch, 
            "activation_fn": activation_fn, 
            "ortho_init": ortho_init, 
            "optimizer_class": RMSpropTFLike, 
            "optimizer_kwargs": dict(alpha=0.99, eps=1e-5, weight_decay=0) 
        }, 
    } 


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: VecEnv,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:

    # Sample hyperparameters 
    kwargs = sample_a2c_params(trial) 
    # Create env 
    env = make_vec_env(lambda: GBR_v0(local=False, screen_size=500),
                        n_envs=N_ENVS, 
                        vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env)

    # Create the RL model 
    model = A2C(policy="MlpPolicy", 
                env=env,
                device='cpu', 
                **kwargs) 
    
    # Create env used for evaluation 
    eval_env = make_vec_env(lambda: GBR_v0(local=False, screen_size=500),
                        n_envs=N_ENVS, 
                        vec_env_cls=SubprocVecEnv)
    eval_env = VecNormalize(eval_env)

    # Create the callback that will periodically evaluate
    # and report the performance
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)  # // N_ENVS  
    except (AssertionError, ValueError) as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float('nan')

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training
    # torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    study.trials_dataframe().to_csv(f"{directory}study_results_a2c.csv")

    with open(f"{directory}study.pkl", "wb+") as f:
        pkl.dump(study, f)

    # fig1 = plot_optimization_history(study)
    # fig2 = plot_param_importances(study)

    # fig1.show()
    # fig2.show()

