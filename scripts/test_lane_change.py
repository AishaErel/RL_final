from pathlib import Path
import sys

# Add the src/ folder to Python's path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from rl_final.lane_change_env import LaneChangeEnv


def main():
    print("Creating environment...")

    env = LaneChangeEnv(
        traffic_density="low",
        communication_cost=0.05,
        render_mode="rgb_array",
    )

    print(
        "HighwayEnv action mapping:",
        env.base_env.unwrapped.action_type.actions_indexes,
    )

    observation, info = env.reset(seed=42)

    print("\nInitial observation:")
    print(observation)

    for step in range(15):
        # Maintain speed briefly, then attempt a lane change.
        if step < 2:
            drive_action = 0
        else:
            drive_action = 3

        # Use no communication for this first lane-change test.
        request_action = 0

        action = {
            "drive": drive_action,
            "request": request_action,
        }

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        print(f"\nStep: {step}")
        print("Drive action:", drive_action)
        print("Request action:", request_action)
        print("Reward:", reward)
        print("Current lane:", info["current_lane"])
        print("Success:", info["success"])
        print("Collision:", info["collision"])
        print("Front gap:", info["front_gap"])
        print("Rear gap:", info["rear_gap"])
        print("Local observation:", observation["local"])

        if terminated or truncated:
            print("\nEpisode ended.")
            break

    env.close()
    print("\nTest complete.")


if __name__ == "__main__":
    main()
