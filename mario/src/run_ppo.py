import time

t0 = time.time()
print("Starting run_ppo...")

import retro
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# --- Load the same environment setup ---
def make_env():
    env = retro.make(game="SuperMarioBros-Nes", state="Level1-1", render_mode="rgb_array")
    env = WarpFrame(env)
    return env

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

# --- Load trained model ---
model = PPO.load("models/ppo_collector_5_lowergammalambda.zip")

print(f"Done initializing! Took {round(time.time() - t0, 2)}s")
# --- Run it and visualize ---
obs = env.reset()
for _ in range(20000):
    action, _ = model.predict(obs)
    obs, reward, done, infos = env.step(action)

    frame = env.render()
    if frame is not None:
        cv2.imshow("Agent Playing", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    if done.any():
        obs = env.reset()

env.close()
cv2.destroyAllWindows()
