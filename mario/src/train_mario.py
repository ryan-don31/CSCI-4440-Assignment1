# ------------------------------------------------------------------------------------- IMPORTS -------------------------------------------------------------------------------------
import os
import csv
import time
if __name__ == "__main__":
    t = time.time()
    print("Importing packages...")
import retro
import torch
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, VecNormalize
from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from configs.reward_wrappers import SpeedRewardWrapper, CollectorRewardWrapper
if __name__ == "__main__":
    print(f"Packages imported! {round(time.time() - t, 2)}s")
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ***********************************************************************************************************************************************************************************
# ------------------------------------------------------------------------------------- CONFIGS -------------------------------------------------------------------------------------

# ------------------------------------- ALGORITHMS + PERSONAS -------------------------------------
"""
Algorithm options: "PPO", "A2C"

Persona options:
"speedrunner" - Prioritizes going to the right, beating the level
"collector" - Priotitizes getting coins
"""


# ---------------------------------------- SAVE/LOAD PATHS ----------------------------------------
"""
Defines the names of the model, log, and env files:

Entering an existing model name for LOAD_MODEL_NAME will load that model and continue training it
Entering a new model name will create a new model with that name

Change SAVE_MODEL_NAME to create a branch/copy of an existing model, saving it to a seperate file
Leave SAVE_MODEL_NAME equal to LOAD_MODEL_NAME otherwise

WARNING: Loading a model and changing the SELECTED_ALGO to one different than what the model is
originally trained on causes issues. I don't feel like fixing this right now so just please don't
"""
# -------------------------------------- ENVIRONMENT CONFIGS --------------------------------------
"""
Super straightforward

total_timesteps -> how many timesteps the environment will train for
num_envs -> how many environments will train at the same time
headless -> False = see mario training in real time, True = no visuals, way faster training
SEED -> Random seed for training
"""


# --------- Modify These ---------

SELECTED_ALGO = "PPO"
SELECTED_PERSONA = "collector"

LOAD_MODEL_NAME = "ppo_collector"
SAVE_MODEL_NAME = LOAD_MODEL_NAME

total_timesteps = 1_000_000
num_envs = 8
headless = True
SEED = 42

# --------------------------------


# --- Error checks for choosing unsupported algos or personas ---
if SELECTED_ALGO not in ("PPO", "A2C"):
    print("selected_algo must be \"PPO\" or \"A2C\"")

if SELECTED_PERSONA not in ("speedrunner", "collector"):
    print("selected_persona must be \"speedrunner\" or \"collector\"")

# -------------------------- SEEDING -----------------------------
# - Got the code for this from Sid (Doom environment in this repo)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_random_seed(SEED)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ***********************************************************************************************************************************************************************************
# ----------------------------------------------------------------------------------- CSV LOGGER ------------------------------------------------------------------------------------
class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, log_dir="../logs", log_file=f"{SAVE_MODEL_NAME}.csv", verbose=1):
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
    

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ***********************************************************************************************************************************************************************************
# -------------------------------------------------------------------------------- HELPER FUNCTION ----------------------------------------------------------------------------------
# --- Applies wrappers to the game environment. Used in the main function
def make_env(game, state, render=False):
    def _init():
        print(f"[DEBUG] Creating environment in PID {os.getpid()} | render={render}")
        env = retro.make(game=game, state=state, render_mode="human" if render else None)
        env = WarpFrame(env)
        env = MaxAndSkipEnv(env, skip=4)
        if SELECTED_PERSONA == "speedrunner":
            env = SpeedRewardWrapper(env)
        elif SELECTED_PERSONA == "collector":
            env = CollectorRewardWrapper(env)
        env = Monitor(env)
        return env
    return _init

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ***********************************************************************************************************************************************************************************
# --------------------------------------------------------------------------------- MAIN FUNCTION -----------------------------------------------------------------------------------
def main():
    # -- Config setup
    game = "SuperMarioBros-Nes"
    state = "Level1-1"
    load_model_path = f"models/{LOAD_MODEL_NAME}.zip"
    load_venv_path = f"envs/{LOAD_MODEL_NAME}.pkl"

    save_model_path = f"models/{SAVE_MODEL_NAME}.zip"
    save_venv_path = f"envs/{SAVE_MODEL_NAME}.pkl"

    if num_envs > 1:
        env_fns = [make_env(game, state, render=(not headless)) for _ in range(num_envs)]
    else:
        env_fns = [make_env(game, state, render=(not headless))] # Stinky code, "not headless" is the inversion of the headless bool from earlier. if headless=False then render=True

    # Initialize vec environment
    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv(env_fns), n_stack=4))

    # ------------------- LOADING ENVIRONMENT AND MODEL -------------------
    # --- Try to load both, otherwise create a new one
    if os.path.exists(load_venv_path):
        print(f"Loading existing training env from {load_venv_path}")
        venv = VecNormalize.load(load_venv_path, venv)
    else:
        print(f"Creating new training env at {load_venv_path}")
        venv = VecTransposeImage(VecFrameStack(SubprocVecEnv(env_fns), n_stack=4))
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True)

    if os.path.exists(load_model_path):
        print(f"Loading existing model from {load_model_path}")
        if SELECTED_ALGO == "PPO":
            model = PPO.load(load_model_path, env=venv)
        elif SELECTED_ALGO == "A2C":
            model = A2C.load(load_model_path, env=venv)
            model.learning_rate = 2.5e-4
    else:
        print(f"Creating new model at {load_model_path}")
        if SELECTED_ALGO == "PPO":
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
                tensorboard_log="logs/ppo_tensorboard_log/"
            )
        elif SELECTED_ALGO == "A2C":
            model = A2C(
                "CnnPolicy",
                env=venv,
                verbose=1,
                learning_rate=5e-4,
                n_steps=32,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_rms_prop=True,
                tensorboard_log="logs/a2c_tensorboard_log/",
                normalize_advantage=True,
            )

    # -- Initialize callback for logger
    callback = EpisodeLoggerCallback(log_dir=f"logs/new-tests/{SELECTED_ALGO}/{SELECTED_PERSONA}", verbose=1)

    # Train
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps, 
        reset_num_timesteps=False, 
        callback=callback)
    
    model.save(save_model_path)
    venv.save(save_venv_path)
    venv.close()

if __name__ == "__main__":
    main()
