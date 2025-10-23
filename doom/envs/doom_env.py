from vizdoom import DoomGame, GameVariable
import numpy as np
import cv2
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import time
from scipy.stats import norm
import os
import csv
from stable_baselines3.common.callbacks import BaseCallback


class DoomDefendCenterLoggerCallback(BaseCallback):
    """
    Logs per-episode metrics for Defend the Center.
    Tracks total_reward, episode length, kill count, and health loss rate.
    """
    def __init__(self, log_dir="logs", log_file="defend_center.csv", verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self.episode_counter = 0
        self.episode_health_start = {}  # track starting health per env

        # CSV setup
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode", "total_reward", "length",
                    "killcount", "avg_health_loss_per_step",
                    "total_timesteps"
                ])
        else:
            with open(self.log_path, "r") as f:
                lines = list(csv.reader(f))
                self.episode_counter = len(lines) - 1

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            info = infos[i]

            # record starting health if first step
            if i not in self.episode_health_start:
                self.episode_health_start[i] = info.get("health", 100)

            if done:
                ep_rew = info.get("episode", {}).get("r", 0.0)
                ep_len = info.get("episode", {}).get("l", 0)
                kills = info.get("killcount", 0)

                start_health = self.episode_health_start.pop(i, 100)
                end_health = info.get("health", 0)
                avg_health_loss_per_step = (start_health - end_health) / max(ep_len, 1)

                self.episode_counter += 1

                # append to CSV
                with open(self.log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.episode_counter,
                        ep_rew,
                        ep_len,
                        kills,
                        avg_health_loss_per_step,
                        self.num_timesteps
                    ])

                if self.verbose > 0:
                    print(f"[DefendCenter] Episode {self.episode_counter} | Reward: {ep_rew} | Len: {ep_len} | Kills: {kills} | Avg health loss/step: {avg_health_loss_per_step:.2f}")

        return True


class DoomEpisodeLoggerCallback(BaseCallback):
    """
    Logs per-episode metrics to CSV for VizDoom.
    Tracks: ammo, ammo_delta, health, hitcount, hitcount_delta, damage_taken, damage_taken_delta
    """

    def __init__(self, log_dir="logs", log_file="doom_episodes.csv", verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self.episode_counter = 0

        # Initialize CSV header if file doesn't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "total_reward",
                    "length",
                    "ammo", "ammo_delta",
                    "health",
                    "hitcount", "hitcount_delta",
                    "damage_taken", "damage_taken_delta",
                    "total_timesteps"
                ])
        else:
            with open(self.log_path, "r") as f:
                self.episode_counter = sum(1 for _ in f) - 1  # subtract header

    def _on_step(self) -> bool:
        # VecEnv returns a list of infos per env
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        for i, info in enumerate(infos):
            if "episode" in info:  # SB3 marks episode ends in info["episode"]
                ep_rew = info["episode"]["r"]
                ep_len = info["episode"]["l"]

                ammo = info.get("ammo", 0)
                ammo_delta = info.get("ammo_delta", 0)
                health = info.get("health", 0)
                hitcount = info.get("hitcount", 0)
                hitcount_delta = info.get("hitcount_delta", 0)
                damage_taken = info.get("damage_taken", 0)
                damage_taken_delta = info.get("damage_taken_delta", 0)

                self.episode_counter += 1

                with open(self.log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.episode_counter,
                        ep_rew,
                        ep_len,
                        ammo, ammo_delta,
                        health,
                        hitcount, hitcount_delta,
                        damage_taken, damage_taken_delta,
                        self.num_timesteps
                    ])

                if self.verbose > 0:
                    print(f"Episode {self.episode_counter} | Reward: {ep_rew:.2f} | Length: {ep_len} | Health: {health}")

        return True

class VizDoomGym(Env):
    """
    Minimal VizDoom Gymnasium wrapper for basic scenario.
    Produces grayscale 100x160 frames and handles ammo as an example info variable.
    """

    def __init__(self, scenario_path, render=True, number_of_actions=3):
        super().__init__()

        self.game = DoomGame()
        self.game.load_config(scenario_path)
        self.game.set_window_visible(render)
        self.game.init()

        # action and observation space
        self.number_of_actions = number_of_actions
        self.action_space = Discrete(number_of_actions)
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)

        self.last_kills = 0

    def step(self, action):
        actions = np.eye(self.number_of_actions, dtype=np.uint8)
        self.game.make_action(actions[action], 4)

        terminated = self.game.is_episode_finished()
        truncated = self.game.is_player_dead()

        if self.game.get_state():
            state = self._process_frame(self.game.get_state().screen_buffer)
            ammo = self.game.get_state().game_variables[0]  # AMMO2
            health = self.game.get_state().game_variables[1]  # HEALTH
            kills = self.game.get_state().game_variables[2]  # KILLCOUNT

            # Reward shaping
            reward = (kills - self.last_kills) * 1.0
            self.last_kills = kills

            # Death penalty
            if truncated:
                reward -= 1.0
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)
            ammo, health, kills = 0, 0, 0
            reward = -1.0 if truncated else 0.0

        info = {"ammo": ammo, "health": health, "killcount": kills}
        return state, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        self.last_kills = 0

        if self.game.get_state():
            state = self._process_frame(self.game.get_state().screen_buffer)
            ammo = self.game.get_state().game_variables[0]
            health = self.game.get_state().game_variables[1]
            kills = self.game.get_state().game_variables[2]
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)
            ammo, health, kills = 0, 0, 0

        return state, {"ammo": ammo, "health": health, "killcount": kills}

    def _process_frame(self, frame):
        gray = cv2.cvtColor(np.moveaxis(frame, 0, -1), cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1)


class VizDoomGymCorridor(Env):
    """
    Custom Gymnasium environment for the VizDoom 'Deadly Corridor' scenario.
    Wraps the DoomGame into a Gym-like API for reinforcement learning.
    """

    def __init__(self, scenario_path, render=True, number_of_actions=7):
        super().__init__()

        # Initialize the VizDoom game
        self.game = DoomGame()
        self.game.load_config(scenario_path)   # Load scenario config (.cfg)
        self.game.set_window_visible(render)   # Show or hide rendering window
        self.game.init()

        # Define Gym spaces
        self.number_of_actions = number_of_actions
        self.action_space = Discrete(number_of_actions)  # 7 discrete actions
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)

        # Track relevant game variables for reward calculation
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 26

    def step(self, action):
        """
        Executes one environment step using the chosen action.
        Returns next_state, reward, terminated, truncated, info.

        For performing reward shaping on the scenario
        """
        actions = np.identity(self.number_of_actions)
        movement_reward = self.game.make_action(actions[action], 4)

        reward = 0
        if self.game.get_state():
            state = self.grayscale(self.game.get_state().screen_buffer)
            health, damage_taken, hitcount, ammo, camera_angle = self.game.get_state().game_variables

            # Compute deltas
            damage_taken_delta = damage_taken - self.damage_taken
            hitcount_delta = hitcount - self.hitcount
            ammo_delta = ammo - self.ammo

            # Update stored values
            self.damage_taken = damage_taken
            self.hitcount = hitcount
            self.ammo = ammo

            # Reward shaping
            camera_reward = -abs(camera_angle - 180) / 180
            reward = (
                movement_reward
                + hitcount_delta * 210
                + ammo_delta * 5
                - damage_taken_delta * 10
                + camera_reward
            )
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)

        # Wrap extra info into a dictionary
        info = {
            "ammo":                 ammo,
            "ammo_delta":           ammo_delta,
            "health":               health,
            "hitcount":             hitcount,
            "hitcount_delta":       hitcount_delta,
            "damage_taken":         damage_taken, 
            "damage_taken_delta":   damage_taken_delta,
        }

        # Episode termination logic
        terminated = self.game.is_episode_finished()
        truncated = self.game.is_player_dead()

        return state, reward, terminated, truncated, info

    def reset(self, seed=0):
        """
        Starts a new episode and returns the initial state and info.
        """
        self.game.new_episode()

        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 26

        if self.game.get_state() is None:
            time.sleep(0.05)

        state = self.grayscale(self.game.get_state().screen_buffer)

        info = {
            "ammo": self.ammo,
            "ammo_delta": 0,
            "health": self.game.get_game_variable(GameVariable.HEALTH),
            "hitcount": self.hitcount,
            "hitcount_delta": 0,
            "damage_taken": self.damage_taken,
            "damage_taken_delta": 0,
        }

        return state, info

    def grayscale(self, observation):
        """
        Converts a Doom RGB frame into grayscale and resizes it.
        """
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state

    def close(self):
        """
        Properly closes the VizDoom game instance.
        """
        self.game.close()

