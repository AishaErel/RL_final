# defines the scripted neighboring behaviors
# maintain speed, accelerate, brake
# they will follow rules only but their futre actions are hidden from RL agent
import numpy as np


class NeighborBehaviorModel:
    """
    Scripted behavior model for neighboring vehicles.

    Behaviors:
        maintain: keep approximately the same speed
        accelerate: increase speed
        brake: decrease speed
    """

    BEHAVIORS = ["maintain", "accelerate", "brake"]

    def __init__(
        self,
        acceleration=2.0,
        braking=3.0,
        maximum_speed=30.0,
        time_step=1.0,
    ):
        self.acceleration = acceleration
        self.braking = braking
        self.maximum_speed = maximum_speed
        self.time_step = time_step

        self.random_generator = np.random.default_rng()
        self.vehicle_behaviors = {}

    def reset(self, vehicles, ego_vehicle, seed=None):
        """
        Assign one hidden scripted behavior to each neighboring vehicle.
        """

        if seed is not None:
            self.random_generator = np.random.default_rng(seed)

        self.vehicle_behaviors = {}

        for vehicle in vehicles:
            if vehicle is ego_vehicle:
                continue

            behavior = self.random_generator.choice(self.BEHAVIORS)
            self.vehicle_behaviors[id(vehicle)] = behavior

    def get_behavior(self, vehicle):
        """
        Return a vehicle's hidden behavior.
        """

        return self.vehicle_behaviors.get(id(vehicle), "maintain")

    def update(self, vehicles, ego_vehicle):
        """
        Update neighboring vehicle speeds according to their
        scripted behaviors.
        """

        for vehicle in vehicles:
            if vehicle is ego_vehicle:
                continue

            behavior = self.get_behavior(vehicle)

            if behavior == "accelerate":
                vehicle.speed = min(
                    vehicle.speed
                    + self.acceleration * self.time_step,
                    self.maximum_speed,
                )

            elif behavior == "brake":
                vehicle.speed = max(
                    vehicle.speed
                    - self.braking * self.time_step,
                    0.0,
                )

            elif behavior == "maintain":
                # Keep the current speed.
                vehicle.speed = vehicle.speed

    def predicted_speeds(self, vehicle, horizon=3):
        """
        Predict the neighbor's future speeds.

        These predictions are sent only if the ego vehicle requests
        information.
        """

        behavior = self.get_behavior(vehicle)
        current_speed = vehicle.speed
        predictions = []

        for step in range(1, horizon + 1):
            if behavior == "accelerate":
                predicted_speed = current_speed + (
                    self.acceleration
                    * self.time_step
                    * step
                )

            elif behavior == "brake":
                predicted_speed = current_speed - (
                    self.braking
                    * self.time_step
                    * step
                )

            else:
                predicted_speed = current_speed

            predicted_speed = np.clip(
                predicted_speed,
                0.0,
                self.maximum_speed,
            )

            predictions.append(float(predicted_speed))

        return predictions

    def create_message(self, vehicle, ego_vehicle):
        """
        Create a compact predicted-state message.
        """

        relative_position = (
            vehicle.position[0] - ego_vehicle.position[0]
        )

        relative_speed = vehicle.speed - ego_vehicle.speed

        predicted_speeds = self.predicted_speeds(
            vehicle,
            horizon=3,
        )

        message = np.array(
            [
                relative_position,
                relative_speed,
                predicted_speeds[0],
                predicted_speeds[1],
                predicted_speeds[2],
            ],
            dtype=np.float32,
        )

        return message
