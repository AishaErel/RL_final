from pathlib import Path
import sys
import importlib.util
import csv
from datetime import datetime


# Locate the project root
project_root = Path(__file__).resolve().parents[1]

# Find lane_change_env.py automatically
matches = list(project_root.rglob("lane_change_env.py"))

if not matches:
    raise FileNotFoundError(
        f"Could not find lane_change_env.py inside {project_root}"
    )

environment_file = matches[0]
sys.path.insert(0, str(environment_file.parent))

# Import LaneChangeEnv directly
spec = importlib.util.spec_from_file_location(
    "lane_change_env",
    environment_file,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

LaneChangeEnv = module.LaneChangeEnv


DENSITIES = ["low", "medium", "high"]
COMMUNICATION_CONDITIONS = {
    "no_communication": 0,
    "ask_front": 1,
}

SEEDS = [0, 1, 2]
EPISODES_PER_SEED = 10
MAX_STEPS = 15


def run_episode(traffic_density, request_action, seed):
    env = LaneChangeEnv(
        traffic_density=traffic_density,
        communication_cost=0.05,
        render_mode="rgb_array",
    )

    observation, info = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    success = False
    collision = False

    for step in range(MAX_STEPS):
        # Maintain speed for two steps, then attempt a lane change.
        if step < 2:
            drive_action = 0
        else:
            drive_action = 3

        action = {
            "drive": drive_action,
            "request": request_action,
        }

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward
        steps += 1

        success = bool(info["success"])
        collision = bool(info["collision"])

        if terminated or truncated:
            break

    env.close()

    return {
        "success": success,
        "collision": collision,
        "reward": total_reward,
        "steps": steps,
    }


def main():
    results = []

    print("Using environment file:")
    print(environment_file)
    print()

    for density in DENSITIES:
        for condition_name, request_action in (
            COMMUNICATION_CONDITIONS.items()
        ):
            episode_results = []

            for seed in SEEDS:
                for episode_number in range(EPISODES_PER_SEED):
                    episode_seed = seed * 1000 + episode_number

                    result = run_episode(
                        traffic_density=density,
                        request_action=request_action,
                        seed=episode_seed,
                    )

                    episode_results.append(result)

            total_episodes = len(episode_results)
            successes = sum(
                result["success"] for result in episode_results
            )
            collisions = sum(
                result["collision"] for result in episode_results
            )

            success_rate = successes / total_episodes
            collision_rate = collisions / total_episodes
            average_reward = sum(
                result["reward"] for result in episode_results
            ) / total_episodes
            average_steps = sum(
                result["steps"] for result in episode_results
            ) / total_episodes

            summary = {
                "traffic_density": density,
                "communication": condition_name,
                "episodes": total_episodes,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
                "average_reward": average_reward,
                "average_steps": average_steps,
            }

            results.append(summary)

            print(
                f"{density:>6} | "
                f"{condition_name:>16} | "
                f"episodes={total_episodes:>3} | "
                f"success={success_rate:.2f} | "
                f"collision={collision_rate:.2f} | "
                f"reward={average_reward:.2f}"
            )

    results_directory = project_root / "results"
    results_directory.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = (
        results_directory
        / f"environment_validation_{timestamp}.csv"
    )

    with open(results_file, "w", newline="") as file:
        fieldnames = [
            "traffic_density",
            "communication",
            "episodes",
            "success_rate",
            "collision_rate",
            "average_reward",
            "average_steps",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Saved results to: {results_file}")


if __name__ == "__main__":
    main()
