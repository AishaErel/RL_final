from pathlib import Path
import sys
import importlib.util
import numpy as np


project_root = Path(__file__).resolve().parents[1]

# Find lane_change_env.py
matches = list(project_root.rglob("lane_change_env.py"))

if not matches:
    raise FileNotFoundError("Could not find lane_change_env.py")

environment_file = matches[0]
sys.path.insert(0, str(environment_file.parent))

# Load LaneChangeEnv
spec = importlib.util.spec_from_file_location(
    "lane_change_env",
    environment_file,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

LaneChangeEnv = module.LaneChangeEnv


def flatten_observation(observation):
    return np.concatenate(
        [
            observation["local"],
            observation["message"],
            np.array(
                [observation["message_received"]],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def main():
    densities = ["low", "medium", "high"]
    episodes_per_density = 30
    max_steps = 15

    observations = []
    actions = []
    next_observations = []
    rewards = []
    terminated_values = []
    density_values = []

    density_to_id = {
        "low": 0,
        "medium": 1,
        "high": 2,
    }

    for density in densities:
        env = LaneChangeEnv(
            traffic_density=density,
            communication_cost=0.05,
            render_mode="rgb_array",
        )

        for episode in range(episodes_per_density):
            seed = density_to_id[density] * 1000 + episode

            observation, info = env.reset(seed=seed)

            for step in range(max_steps):
                # Random actions provide varied training data.
                action = env.action_space.sample()

                next_observation, reward, terminated, truncated, info = (
                    env.step(action)
                )

                observations.append(
                    flatten_observation(observation)
                )

                actions.append(
                    [
                        int(action["drive"]),
                        int(action["request"]),
                    ]
                )

                next_observations.append(
                    flatten_observation(next_observation)
                )

                rewards.append(float(reward))
                terminated_values.append(
                    float(terminated or truncated)
                )
                density_values.append(density_to_id[density])

                observation = next_observation

                if terminated or truncated:
                    break

        env.close()

    data_directory = project_root / "data"
    data_directory.mkdir(exist_ok=True)

    output_file = data_directory / "transitions.npz"

    np.savez_compressed(
        output_file,
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        next_observations=np.array(
            next_observations,
            dtype=np.float32,
        ),
        rewards=np.array(rewards, dtype=np.float32),
        terminated=np.array(
            terminated_values,
            dtype=np.float32,
        ),
        densities=np.array(
            density_values,
            dtype=np.int64,
        ),
    )

    print("Data collection complete.")
    print(f"Saved {len(observations)} transitions.")
    print(f"Saved file: {output_file}")


if __name__ == "__main__":
    main()
