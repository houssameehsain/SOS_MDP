import pickle as pkl 
from typing import Any, Dict 

import os 
# import gym 
# import torch 
import torch.nn as nn 
from stable_baselines3 import DQN 
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.callbacks import EvalCallback 
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from envs import GBR_v0, GBR_v1, GPN_v0 

import optuna 
from optuna.pruners import MedianPruner 
from optuna.samplers import TPESampler 
# from optuna.visualization import plot_optimization_history, plot_param_importances 


N_TRIALS = 1000
N_JOBS = 1
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e5)
N_EVAL_EPISODES = 100
N_ENVS = 12
EVAL_FREQ = max(int(N_TIMESTEPS / N_EVALUATIONS) // N_ENVS, 1)

TIMEOUT = int(60 * 3000)  # 480 minutes

run_id = "hypertune_dqn_brv0_02"
directory = f"./logs/run_{run_id}/"
os.makedirs(directory, exist_ok=True)


def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DQN hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)])
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.001, 0.2, log=True)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.001, 0.5, log=True)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 5000, 10000, 20000])

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
    subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])
    gradient_steps = max(train_freq // subsample_steps, 1)

    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium", "large"])

    net_arch = {"tiny": [64], "small": [64, 64], "medium": [256, 256], "large": [256, 256, 256]}[net_arch]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    return hyperparams


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
    kwargs = sample_dqn_params(trial) 
    # Create env 
    env = make_vec_env(lambda: GBR_v0(local=False, screen_size=500),
                        n_envs=N_ENVS, 
                        vec_env_cls=SubprocVecEnv)
    # env = VecNormalize(env)

    # Create the RL model 
    model = DQN(policy="CnnPolicy", 
                env=env,
                device='cpu', 
                **kwargs) 
    
    # Create env used for evaluation 
    eval_env = make_vec_env(lambda: GBR_v0(local=False, screen_size=500),
                        n_envs=N_ENVS, 
                        vec_env_cls=SubprocVecEnv)
    # eval_env = VecNormalize(eval_env)

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
    study.trials_dataframe().to_csv(f"{directory}study_results_dqn.csv")

    with open(f"{directory}study.pkl", "wb+") as f:
        pkl.dump(study, f)

    # fig1 = plot_optimization_history(study)
    # fig2 = plot_param_importances(study)

    # fig1.show()
    # fig2.show()


