import gymnasium as gym

# ---------------------------------------------- SPEEDRUNNER PERSONA ----------------------------------------------
# - Prioritizes right movement, small penalty for every frame to encourage constant movement + penalize getting stuck
# - HUGE reward for beating a level, encourages b-lining it to the end of the level by any means necessary
class SpeedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_x = 0
        self.last_coins = 0
        self.last_lives = 0
        self.last_time = 400

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_x = (info.get("xscrollHi", 0) << 8) + info.get("xscrollLo", 0)
        self.last_coins = info.get("coins", 0)
        self.last_lives = info.get("lives", 0)
        self.last_time = info.get("time", 400)
        return obs, info

    def step(self, action):
        obs, legacy_reward, terminated, truncated, info = self.env.step(action)

        # -- Calculate value for custom "xscroll" value in the dictionary --
        # -- Capture values for distance travelled this frame --
        info["xscroll"] = (info.get("xscrollHi", 0) << 8) + info.get("xscrollLo", 0)
        x = info["xscroll"]
        dx = x - self.last_x
        self.last_x = x
        # -- Capture current time remaining --
        time_left = info.get("time", self.last_time)
        # -- Capture coins collected this frame --
        coins = info.get("coins", 0)
        coin_diff = coins - self.last_coins
        self.last_coins = coins

        reward = 0.0

        # -- Reward for moving forward --
        # -- We don't want to penalize backward movement, otherwise mario would get stuck --
        if dx > 0:
            reward += dx * 0.4

        # -- Small penalty every frame --
        # -- If mario isn't moving forward, getting coins or beating the level right now then he gets the belt --
        reward -= 0.05

        # -- Small reward for getting coins --
        # -- Don't count negative change to avoid penalty for coin counter resetting --
        if coin_diff > 0:
            reward += coin_diff * 5.0

        # -- Mega reward for beating the level --
        if terminated and info.get("levelHi") < info.get("levelLo"):
            reward += 2000 + time_left * 2

        return obs, reward, terminated, truncated, info

# ----------------------------------------------- COLLECTOR PERSONA -----------------------------------------------
# - Priotizes getting coins and moving to the right
# - Slight reward for getting to the end of the level, no penalty for not moving (to avoid penalizing exploration)
class CollectorRewardWrapper(gym.Wrapper):
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

        # -- Calculate value for custom "xscroll" value in the dictionary --
        # -- Capture values for distance travelled this frame --
        info["xscroll"] = (info.get("xscrollHi", 0) << 8) + info.get("xscrollLo", 0)
        x = info.get("xscroll")
        dx = x - self.last_x
        self.last_x = x
        # -- Capture coins collected this frame --
        coins = info.get("coins", 0)
        coin_diff = coins - self.last_coins
        self.last_coins = coins

        reward = 0.0

        # -- Reward for moving forward --
        # -- We don't want to penalize backward movement, otherwise mario would get stuck --
        if dx > 0:
            reward += dx * 0.2
        
        # -- Pretty big reward for getting coins --
        # -- We want mario to crave those things --
        if coin_diff > 0:
            reward += coin_diff * 50.0 
        
        # -- Big reward for beating the level --
        if terminated and info.get("levelHi") < info.get("levelLo"):
            reward += 500

        return obs, reward, terminated, truncated, info