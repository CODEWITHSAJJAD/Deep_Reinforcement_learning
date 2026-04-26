import numpy as np
import gym
from typing import Optional, Dict
from gym import spaces


class SlipperyGridWorldEnv(gym.Env):
    """
    A grid world environment with slippery tiles similar to Frozen Lake.
    Features:
    - Slippery tiles where actions may not execute as intended
    - Different tile types (normal, slippery, goal, hole)
    - Stochastic movement
    """

    def __init__(self, size: int = 5, slippery_prob: float = 0.3):
        self.size = size
        self.slippery_prob = slippery_prob
        self.NORMAL = 0
        self.SLIPPERY = 1
        self.HOLE = 2
        self.GOAL = 3
        self._map = self._generate_map()
        self._agent_location = np.array([0, 0], dtype=np.int32)
        self._target_location = np.array([size - 1, size - 1], dtype=np.int32)
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "map": spaces.Box(0, 3, shape=(size, size), dtype=int)
        })
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    def _generate_map(self):
        """Generate a random map with different tile types"""
        map_grid = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                if (i == 0 and j == 0) or (i == self.size - 1 and j == self.size - 1):
                    continue
                if np.random.random() < 0.3:
                    map_grid[i, j] = self.SLIPPERY

        # Add some holes (10% of tiles)
        for i in range(self.size):
            for j in range(self.size):
                if (i == 0 and j == 0) or (i == self.size - 1 and j == self.size - 1):
                    continue
                if np.random.random() < 0.1:
                    map_grid[i, j] = self.HOLE

        # Set goal position
        map_grid[self.size - 1, self.size - 1] = self.GOAL

        return map_grid

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "map": self._map
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1),
            "current_tile": self._map[tuple(self._agent_location)]
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._agent_location = np.array([0, 0], dtype=np.int32)
        self._map = self._generate_map()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        current_tile = self._map[tuple(self._agent_location)]

        if current_tile == self.SLIPPERY and np.random.random() < self.slippery_prob:
            action = self.action_space.sample()
        elif current_tile == self.SLIPPERY and np.random.random() < self.slippery_prob * 0.5:

            if action == 0:
                action = np.random.choice([1, 3])
            elif action == 1:
                action = np.random.choice([0, 2])
            elif action == 2:  # Up
                action = np.random.choice([1, 3])
            elif action == 3:  # Left
                action = np.random.choice([0, 2])

        # Get the intended direction
        direction = self._action_to_direction[action]
        new_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # Check what tile we're moving to
        new_tile = self._map[tuple(new_location)]

        # Update agent location if not a hole
        if new_tile != self.HOLE:
            self._agent_location = new_location

        # Determine if episode is done
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False

        # Calculate reward
        if terminated:
            reward = 1.0  # Reached goal
        elif new_tile == self.HOLE:
            reward = -1.0  # Fell in a hole
            terminated = True
        else:
            reward = -0.01  # Small step penalty

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


# Register the environment
gym.register(
    id="gymnasium_env/SlipperyGridWorld-v0",
    entry_point=SlipperyGridWorldEnv,
    kwargs={"size": 5, "slippery_prob": 0.3}
)

# Example usage
if __name__ == "__main__":
    env = gym.make("gymnasium_env/SlipperyGridWorld-v0")
    print("Observation space:", env.observation_space)

    obs, info = env.reset()
    print("Initial observation:", obs)
    print("Initial info:", info)

    # Take a random action
    action = env.action_space.sample()
    print("Taking action:", action)

    obs, reward, terminated, truncated, info = env.step(action)
    print("New observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Info:", info)