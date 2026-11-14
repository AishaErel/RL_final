import gymnasium as gym
from gymnasium import spaces
import numpy as np
import highway_env


class LaneChangeEnv(gym.Env):
  
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        traffic_density="medium",
        communication_cost=0.05,
        render_mode="rgb_array",
    ):
        super().__init__()

        density_settings = {
            "low": {
                "vehicles_count": 10,
                "vehicles_density": 0.5,
            },
            "medium": {
                "vehicles_count": 25,
                "vehicles_density": 1.0,
            },
            "high": {
                "vehicles_count": 50,
                "vehicles_density": 1.5,
            },
        }

        if traffic_density not in density_settings:
            raise ValueError(
                "traffic_density must be 'low', 'medium', or 'high'"
            )

        self.traffic_density = traffic_density
        self.communication_cost = communication_cost
        self.render_mode = render_mode

        config = {
            "lanes_count": 2,
            "vehicles_count": density_settings[traffic_density]["vehicles_count"],
            "vehicles_density": density_settings[traffic_density]["vehicles_density"],
            "initial_lane_id": 0,
            "duration": 40,
            "offroad_terminal": True,
        }

        self.base_env = gym.make(
            "highway-v0",
            config=config,
            render_mode=render_mode,
        )

        # Driving action:
        # 0 = maintain speed
        # 1 = accelerate
        # 2 = brake
        # 3 = initiate lane change
        #
        # Request action:
        # 0 = no request
        # 1 = ask front vehicle
        # 2 = ask rear vehicle
        
        self.action_space = spaces.Dict(
            {
                "drive": spaces.Discrete(4),
                "request": spaces.Discrete(3),
            }
        )

        # Local observation:
        # [ego_speed,
        #  current_lane,
        #  front_gap,
        #  rear_gap,
        #  front_relative_speed,
        #  rear_relative_speed]
        #
        # Message:
        # [relative_position,
        #  relative_speed,
        #  predicted_speed_t1,
        #  predicted_speed_t2,
        #  predicted_speed_t3]
        self.observation_space = spaces.Dict(
            {
                "local": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(6,),
                    dtype=np.float32,
                ),
                "message": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(5,),
                    dtype=np.float32,
                ),
                "message_received": spaces.Discrete(2),
            }
        )

        self.start_lane = 0
        self.target_lane = 1
        self.minimum_gap = 10.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        _, info = self.base_env.reset(seed=seed)

        self.start_lane = 0
        self.target_lane = 1

        observation = {
            "local": self._get_local_observation(),
            "message": np.zeros(5, dtype=np.float32),
            "message_received": 0,
        }

        return observation, info

    def step(self, action):
        drive_action = int(action["drive"])
        request_action = int(action["request"])

        # Communication happens before the driving action.
        if request_action == 0:
            message = np.zeros(5, dtype=np.float32)
            message_received = 0

        elif request_action == 1:
            message, found = self._get_neighbor_message("front")
            message_received = int(found)

        elif request_action == 2:
            message, found = self._get_neighbor_message("rear")
            message_received = int(found)

        else:
            raise ValueError("Invalid request action")

        # Check the local situation before the driving action.
        local_observation_before = self._get_local_observation()

        front_gap = local_observation_before[2]
        rear_gap = local_observation_before[3]

        unsafe_lane_change = (
            drive_action == 3
            and (
                front_gap < self.minimum_gap
                or rear_gap < self.minimum_gap
            )
        )

        # Convert our driving action into a HighwayEnv action.
        highway_action = self._convert_drive_action(drive_action)

        _, _, base_terminated, base_truncated, info = self.base_env.step(
            highway_action
        )

        ego = self.base_env.unwrapped.vehicle
        current_lane = ego.lane_index[2]

        collision = bool(ego.crashed)
        success = current_lane == self.target_lane and not collision

        # Driving reward.
        reward = -0.01

        if drive_action == 1:
            reward += 0.01

        if unsafe_lane_change:
            reward -= 2.0

        if collision:
            reward -= 10.0

        if success:
            reward += 10.0

        # Communication cost.
        if request_action != 0:
            reward -= self.communication_cost

        terminated = bool(
            base_terminated
            or collision
            or success
        )

        truncated = bool(base_truncated)

        observation = {
            "local": self._get_local_observation(),
            "message": message,
            "message_received": message_received,
        }

        info.update(
            {
                "success": success,
                "collision": collision,
                "traffic_density": self.traffic_density,
                "drive_action": drive_action,
                "request_action": request_action,
                "communication_cost": (
                    self.communication_cost
                    if request_action != 0
                    else 0.0
                ),
            }
        )

        return observation, reward, terminated, truncated, info

    def _convert_drive_action(self, drive_action):
        """
        Convert our four driving actions to HighwayEnv actions.
        """

        action_indexes = (
            self.base_env.unwrapped.action_type.actions_indexes
        )

        if drive_action == 0:
            return action_indexes["IDLE"]

        if drive_action == 1:
            return action_indexes["FASTER"]

        if drive_action == 2:
            return action_indexes["SLOWER"]

        if drive_action == 3:
            # In this setup, lane 0 is the start lane
            # and lane 1 is the target lane.
            return action_indexes["LANE_LEFT"]

        raise ValueError("Invalid driving action")

    def _get_local_observation(self):
        """
        Return only locally observable information.

        This does not include the neighboring vehicles'
        future behavior or planned trajectory.
        """

        ego = self.base_env.unwrapped.vehicle

        ego_speed = ego.speed
        current_lane = ego.lane_index[2]
        ego_x = ego.position[0]

        front_gap = 100.0
        rear_gap = 100.0
        front_relative_speed = 0.0
        rear_relative_speed = 0.0

        for vehicle in self.base_env.unwrapped.road.vehicles:
            if vehicle is ego:
                continue

            vehicle_lane = vehicle.lane_index[2]

            # Only inspect vehicles in the target lane.
            if vehicle_lane != self.target_lane:
                continue

            gap = vehicle.position[0] - ego_x
            relative_speed = vehicle.speed - ego.speed

            if gap > 0 and gap < front_gap:
                front_gap = gap
                front_relative_speed = relative_speed

            elif gap < 0 and abs(gap) < rear_gap:
                rear_gap = abs(gap)
                rear_relative_speed = relative_speed

        return np.array(
            [
                ego_speed,
                float(current_lane),
                front_gap,
                rear_gap,
                front_relative_speed,
                rear_relative_speed,
            ],
            dtype=np.float32,
        )

    def _find_neighbor(self, neighbor_type):
        """
        Find the closest front or rear vehicle
        in the target lane.
        """

        ego = self.base_env.unwrapped.vehicle
        ego_x = ego.position[0]

        closest_front = None
        closest_rear = None
        smallest_front_gap = float("inf")
        smallest_rear_gap = float("inf")

        for vehicle in self.base_env.unwrapped.road.vehicles:
            if vehicle is ego:
                continue

            if vehicle.lane_index[2] != self.target_lane:
                continue

            gap = vehicle.position[0] - ego_x

            if gap > 0 and gap < smallest_front_gap:
                closest_front = vehicle
                smallest_front_gap = gap

            elif gap < 0 and abs(gap) < smallest_rear_gap:
                closest_rear = vehicle
                smallest_rear_gap = abs(gap)

        if neighbor_type == "front":
            return closest_front

        if neighbor_type == "rear":
            return closest_rear

        raise ValueError("neighbor_type must be 'front' or 'rear'")

    def _get_neighbor_message(self, neighbor_type):
        """
        Return a compact predicted-state message.

        This currently uses a simple constant-speed prediction.
        Later, replace it with scripted neighbor behaviors
        such as maintain, accelerate, or brake.
        """

        ego = self.base_env.unwrapped.vehicle
        neighbor = self._find_neighbor(neighbor_type)

        if neighbor is None:
            return np.zeros(5, dtype=np.float32), False

        relative_position = neighbor.position[0] - ego.position[0]
        relative_speed = neighbor.speed - ego.speed

        # Temporary prediction.
        # This will later be generated by the neighbor model.
        predicted_speed_t1 = neighbor.speed
        predicted_speed_t2 = neighbor.speed
        predicted_speed_t3 = neighbor.speed

        message = np.array(
            [
                relative_position,
                relative_speed,
                predicted_speed_t1,
                predicted_speed_t2,
                predicted_speed_t3,
            ],
            dtype=np.float32,
        )

        return message, True

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()