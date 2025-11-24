"""
LQR waypoint tracker example

Author: Hongrui Zheng
Last Modified: 5/5/22
"""

import numpy as np
import gymnasium as gym
from f1tenth_planning.control import LQRController


def main():
    """
    LQR example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # create environment
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "kinematic_state"},
        },
        render_mode="human",
    )

    # create controller
    planner = LQRController(env.unwrapped.track)

    env.unwrapped.add_render_callback(planner.render_waypoints)
    env.unwrapped.add_render_callback(planner.render_local_plan)
    env.unwrapped.add_render_callback(planner.render_control_solution)

    # reset environment
    poses = np.array(
        [
            [
                env.unwrapped.track.raceline.xs[0],
                env.unwrapped.track.raceline.ys[0],
                env.unwrapped.track.raceline.yaws[0],
            ]
        ]
    )
    obs, info = env.reset(options={"poses": poses})
    done = False
    env.render()

    # run simulation
    laptime = 0.0
    while not done:
        steer, speed = planner.plan(
            obs["agent_0"],
        )
        obs, timestep, terminated, truncated, infos = env.step(
            np.array([[steer, speed]])
        )
        done = terminated or truncated
        laptime += timestep
        env.render()
    print("Sim elapsed time:", laptime)


if __name__ == "__main__":
    main()
