# ------------------------------------------------------------------------------------- IMPORTS -------------------------------------------------------------------------------------
import time
import cv2
if __name__ == "__main__":
    t = time.time()
    print("Importing packages...")
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, VecNormalize
from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv
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


# --------- Modify These ---------

MODEL_NAME = "ppo_speedrunner"
SELECTED_ALGO = "PPO"
SELECTED_PERSONA = "speedrunner"

# --------------------------------


if SELECTED_ALGO not in ("PPO", "A2C"):
    print("selected_algo must be \"PPO\" or \"A2C\"")

if SELECTED_PERSONA not in ("speedrunner", "collector"):
    print("selected_persona must be \"speedrunner\" or \"collector\"")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ***********************************************************************************************************************************************************************************
# -------------------------------------------------------------------------------- HELPER FUNCTION ----------------------------------------------------------------------------------
# --- Applies wrappers to the game environment. Used in the main function
def make_env():
    env = retro.make(game="SuperMarioBros-Nes", state="Level1-1", render_mode="rgb_array")
    env = WarpFrame(env)
    env = MaxAndSkipEnv(env, skip=4)
    if SELECTED_PERSONA == "speedrunner":
        env = SpeedRewardWrapper(env)
    elif SELECTED_PERSONA == "collector":
        env = CollectorRewardWrapper(env)
    return env

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ***********************************************************************************************************************************************************************************
# --------------------------------------------------------------------------------- MAIN FUNCTION -----------------------------------------------------------------------------------
def main():
    VENVS_PATH = f"envs/{MODEL_NAME}.pkl"
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # --- Load VecNormalize stats ---
    env = VecNormalize.load(VENVS_PATH, env)
    env.training = False
    env.norm_reward = False

    # --- Load model ---
    model = PPO.load(f"models/{MODEL_NAME}.zip", env=env)

    print("Environment and model loaded. Starting playback...")

    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)


        frame = env.render()
        if frame is not None:
            cv2.imshow("Agent Playing", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break

        if done.any():
            obs = env.reset()

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()