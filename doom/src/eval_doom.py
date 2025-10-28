import argparse
import time
import os
from stable_baselines3 import PPO, A2C
from vizdoom import ScreenResolution
from envs.doom_env import VizDoomGym, VizDoomGymCorridor


# Base directories
MODEL_BASE_DIR = "../models/"
SCENARIO_BASE_DIR = "../configs/scenarios/"

def load_model(algo: str, scenario: str, timesteps: int):
    """
    Load the trained model based on algorithm, scenario, and timesteps.
    """
    model_dir = os.path.join(MODEL_BASE_DIR, algo)
    model_name = f"{algo}_{scenario}_{timesteps}_final.zip"
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


def evaluate(model, scenario, episodes=10, render=True, resolution="1280x960"):
    """Evaluate a trained Doom agent."""


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained VizDoom agent.")

    parser.add_argument("--algo", type=str, choices=["ppo", "a2c"], default="ppo",
                        help="Algorithm used to train (ppo or a2c)")
    parser.add_argument("--scenario", type=str, default="defend_the_center",
                        help="Scenario name (defend_the_center, deadly_corridor)")
    parser.add_argument("--timesteps", type=int, default=125000,
                        help="Training step count used in filename (default: 125000)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--headless", action="store_true",
                        help="Disable rendering for faster evaluation (headless mode)")

    args = parser.parse_args()

    scenario_path = os.path.join(SCENARIO_BASE_DIR, f"{args.scenario}.cfg")


if __name__ == "__main__":
    main()
