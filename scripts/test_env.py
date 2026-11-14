"""
Create the environment.
Reset it.
Print the observation.
Take random driving and request actions.
Check whether the episode ends after success or collision.
Render or plot several scenarios.
"""
import gymnasium as gym
import highway_env

env = gym.make("highway-v0")
obs, info = env.reset()

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Initial observation:", obs)

env.close()