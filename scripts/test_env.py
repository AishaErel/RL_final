from lane_change_env import LaneChangeEnv

env = LaneChangeEnv(
    traffic_density="medium",
    communication_cost=0.05,
)

observation, info = env.reset()

print("Initial observation:")
print(observation)

for step in range(10):
    action = {
        "drive": 0,     # maintain speed
        "request": 1,   # ask front vehicle
    }

    observation, reward, terminated, truncated, info = env.step(action)

    print("Step:", step)
    print("Reward:", reward)
    print("Observation:", observation)

    if terminated or truncated:
        break

env.close()
