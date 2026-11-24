from pathlib import Path
import sys
import importlib.util


# Find the project folder
project_root = Path(__file__).resolve().parents[1]

# Search for lane_change_env.py anywhere inside the project
matches = list(project_root.rglob("lane_change_env.py"))

if not matches:
    raise FileNotFoundError(
        f"Could not find lane_change_env.py inside {project_root}"
    )

environment_file = matches[0]

# Allow the environment file to find related files
sys.path.insert(0, str(environment_file.parent))

# Load lane_change_env.py directly
spec = importlib.util.spec_from_file_location(
    "lane_change_env",
    environment_file,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

LaneChangeEnv = module.LaneChangeEnv


def main():
    print("Using environment file:")
    print(environment_file)

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
        if step < 2:
            drive_action = 0      # maintain speed
        else:
            drive_action = 3      # initiate lane change

        request_action = 1        # no communication for this test

        action = {
            "drive": drive_action,
            "request": request_action,
        }

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        print(f"\nStep: {step}")
        print("Reward:", reward)
        print("Current lane:", info["current_lane"])
        print("Success:", info["success"])
        print("Collision:", info["collision"])
        print("Front gap:", info["front_gap"])
        print("Rear gap:", info["rear_gap"])
        print("Message received:", observation["message_received"])
        print("Message:", observation["message"])

        if terminated or truncated:
            print("\nEpisode ended.")
            break

    env.close()


if __name__ == "__main__":
    main()
