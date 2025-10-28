import time
import os
import yaml
import sys
from stable_baselines3 import PPO, A2C
sys.path.append('..') # make sure files dont violate this convention
from envs.doom_env import VizDoomGym, VizDoomGymCorridor

# Base directories
MODEL_BASE_DIR =  os.path.normpath("../models/")
SCENARIO_BASE_DIR = os.path.normpath("../configs/scenarios/")

with open("../configs/eval.yaml", "r") as f:
    config = yaml.safe_load(f)

ALGORITHM = config["algo"].lower()
SCENARIO = config["scenario"].lower()
EPISODES = config["episodes"]
TIMESTEPS = config["timesteps"]


def load_model(algo: str, scenario: str, timesteps=TIMESTEPS):
    """
    Load the trained model based on algorithm, scenario, and timesteps.
    """
    if scenario == "defend_the_center":
        model_dir = os.path.join(MODEL_BASE_DIR, f"{algo}_defend_center")
        model_name = f"{algo}_defend_center_{timesteps}_final.zip"
    elif scenario == "deadly_corridor":
        model_dir = os.path.join(MODEL_BASE_DIR, f"{algo}_deadly_corridor")
        model_name = f"{algo}_deadly_corridor_{timesteps}_final.zip"
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    model_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    print(f"Loading model: {model_path}")
    
    if algo == "ppo":
        return PPO.load(model_path)
    elif algo == "a2c":
        return A2C.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")


def evaluate(model, scenario, episodes=EPISODES, render=True):
    """Evaluate a trained Doom agent."""
    if scenario == "defend_the_center":
        scenario_path = os.path.join(SCENARIO_BASE_DIR, "defend_the_center.cfg")
        print(f"Loading scenario: {scenario_path}")
        env = VizDoomGym(scenario_path=scenario_path,render=render)
    elif scenario == "deadly_corridor":
        scenario_path = os.path.join(SCENARIO_BASE_DIR, "deadly_corridor.cfg")
        print(f"Loading scenario: {scenario_path}")
        env = VizDoomGymCorridor(scenario_path=scenario_path,render=render)
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    # Run evaluation episodes
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            time.sleep(0.02 if render else 0)
        print(f"Episode {ep + 1}: total_reward = {total_reward:.2f}")

    env.close()
    print(f"Evaluation of {ALGORITHM} in {SCENARIO} has been completed!")


def main():
    model = load_model(ALGORITHM, SCENARIO)
    evaluate(model, SCENARIO)


if __name__ == "__main__":
    main()
