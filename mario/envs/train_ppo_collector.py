import retro
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, VecNormalize
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv, MaxAndSkipEnv
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import csv

model_name = "ppo_collector_5_lowergammalambda"

class EpisodeLoggerCallback(BaseCallback):
    """
    Logs per-episode metrics to a CSV file and appends to existing logs across runs.
    """
    def __init__(self, log_dir="logs", log_file=f"{model_name}.csv", verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self.episode_counter = 0  # total episodes seen so far

        # Initialize CSV header if file doesn't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "total_reward",
                    "length",
                    "xscroll",
                    "coins",
                    "lives",
                    "score",
                    "time_left",
                    "total_timesteps"
                ])
        else:
            # Read current file to continue episode numbering
            with open(self.log_path, "r") as f:
                reader = csv.reader(f)
                lines = list(reader)
                self.episode_counter = len(lines) - 1  # subtract header

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                ep_len = info["episode"]["l"]

                # Pull custom metrics if available
                xscroll = info.get("xscroll", None)
                coins = info.get("coins", None)
                lives = info.get("lives", None)
                score = info.get("score", None)
                time_left = info.get("time", None)

                self.episode_counter += 1

                # Append row to CSV
                with open(self.log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.episode_counter,
                        ep_rew,
                        ep_len,
                        xscroll,
                        coins,
                        lives,
                        score,
                        time_left,
                        self.num_timesteps
                    ])

                if self.verbose > 0:
                    print(f"Episode {self.episode_counter} | Reward: {ep_rew:.2f} | Len: {ep_len} | xscroll: {xscroll}")
        return True


# --- Custom Reward Wrapper ---
class RewardWrapper(gym.Wrapper):
    def __init__(self, env, history_length=5):
        super().__init__(env)
        self.last_x = 0
        self.progress_history = []
        self.history_length = history_length
        self.last_coins = 0
        self.last_lives = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_x = (info.get("xscrollHi", 0) << 8) + info.get("xscrollLo", 0)
        self.progress_history = []
        self.last_coins = info.get("coins", 0)
        self.last_lives = info.get("lives", 0)
        return obs, info

    def step(self, action):
        obs, legacy_reward, terminated, truncated, info = self.env.step(action)

        reward = 0.0

        # -- Saving xscroll to the info dict so it isn't lost later --
        info["xscroll"] = (info.get("xscrollHi", 0) << 8) + info.get("xscrollLo", 0)
        x = info.get("xscroll")
        dx = x - self.last_x
        self.last_x = x
        reward += dx * 0.2

        # -- Saving lives to the info dict so it isn't lost later
        current_lives = info.get("lives", -1)
        if current_lives == -1 and hasattr(self, "last_lives"):
            current_lives = self.last_lives
        else:
            self.last_lives = current_lives
        info["lives"] = current_lives

        # --- Coin Collection Reward ---
        coins = info.get("coins", 0)
        coin_diff = coins - self.last_coins
        if coin_diff > 0:
            reward += coin_diff * 50.0 
        self.last_coins = coins

        # --- Death penalty ---
        # current_lives = info.get("lives", self.last_lives)
        # if current_lives < self.last_lives:
        #     reward -= 50  # slight punishment for dying
        # self.last_lives = current_lives

        if terminated and info.get("levelHi") == 1 and info.get("levelLo") == 2:
            reward += 500  # big bonus for reaching next level

        return obs, reward, terminated, truncated, info


# --- Helper: build one env instance ---
def make_env(game, state):
    def _init():
        env = retro.make(game=game, state=state, render_mode=None)
        env = WarpFrame(env)
        # env = TimeLimit(env, max_episode_steps=500)
        # env = ClipRewardEnv(env)     # clip rewards to [-1, 0, 1]
        env = MaxAndSkipEnv(env, skip=4)
        env = RewardWrapper(env)
        env = Monitor(env)
        return env
    return _init


# --- Main entry point ---
def main():
    game = "SuperMarioBros-Nes"
    state = "Level1-1"
    num_envs = 8
    total_timesteps = 300_000
    model_path = f"models/{model_name}.zip"
    venv_path = f"venvs/{model_name}.pkl"

    # Create vectorized environment (8 parallel workers)
    env_fns = [make_env(game, state) for _ in range(num_envs)]
    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv(env_fns), n_stack=4))

    if os.path.exists(venv_path):
        print(f"Loading existing venv from {venv_path}")
        venv = VecNormalize.load(venv_path, venv)
    else:
        venv = VecTransposeImage(VecFrameStack(SubprocVecEnv(env_fns), n_stack=4))
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True)

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = PPO.load(model_path, env=venv)
        model.learning_rate = 2.5e-4
    else:
        # Create PPO model
        model = PPO(
            "CnnPolicy",
            env=venv,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=2048,
            batch_size=1024,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="ppo_tensorboard_log/"
        )

    callback = EpisodeLoggerCallback(log_dir="ppo_logs", verbose=1)

    # Train
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps, 
        reset_num_timesteps=False, 
        callback=callback)
    
    model.save(model_path)
    venv.save(venv_path)
    venv.close()


# --- Guard for multiprocessing ---
if __name__ == "__main__":
    main()
