import numpy as np
import gym
from typing import Optional
from gym.wrappers import FlattenObservation
class BandedSlipperyWalkEnv(gym.Env):
    def __init__(self, size: int = 5, slip_prob: float = 0.3, band_type: str = "row", band_index: int = 2):
        self.size = size
        self.slip_prob = slip_prob
        self.band_type = band_type
        self.band_index = band_index
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int)
        })
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._agent_location = self.np_random.integers(0, self.size, size=2)
        self._target_location = self._agent_location
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        in_band = False
        if self.band_type == "row" and self._agent_location[1] == self.band_index:
            in_band = True
        elif self.band_type == "column" and self._agent_location[0] == self.band_index:
            in_band = True
        if in_band and self.np_random.random() < self.slip_prob:
            action = self.np_random.integers(0, 4)

        direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info
gym.register(id="gymnasium_env/BandedSlipperyWalk-v0", entry_point=BandedSlipperyWalkEnv)
banded_env = gym.make('gymnasium_env/BandedSlipperyWalk-v0', size=5, slip_prob=0.3, band_type="row", band_index=2)
print("Banded Slippery Walk Observation Space:", banded_env.observation_space)
obs, info = banded_env.reset()
print("Reset:", obs)
wrapped_slippery = FlattenObservation(banded_env)
print("Flattened Observation Space:", wrapped_slippery.observation_space)
print("Flattened Reset:", wrapped_slippery.reset())