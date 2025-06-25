import os
import time

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs import F110Env
from f1tenth_gym.envs.track import Track
from f1tenth_planning.control import Nonlinear_Dynamic_MPPI_Planner
from f1tenth_planning.control.config.controller_config import dynamic_mppi_config
from f1tenth_planning.control.config.dynamics_config import (
    f1fifth_params,
)


def main():
    """
    STMPC example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # create environment
    env: F110Env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg_blank",
            "num_agents": 1,
            "control_input": "accl",
            "observation_config": {"type": "dynamic_state"},
            "params": F110Env.f1fifth_vehicle_params(),
        },
        render_mode="human",
    )

    # Load track waypoints
    waypoints_track: Track = Track.from_raceline_file(
        os.path.join(os.path.dirname(__file__), "trajectory_log.csv"),
        delimiter=";",
        skip_rows=3,
    )

    # create planner
    config = dynamic_mppi_config()
    config.Q = np.array([25.0, 25.0, 0.0, 1.0, 0.1, 0.0, 0.0])
    planner = Nonlinear_Dynamic_MPPI_Planner(track=waypoints_track, params=f1fifth_params())
    env.unwrapped.add_render_callback(planner.render_waypoints)
    env.unwrapped.add_render_callback(planner.render_local_plan)
    env.unwrapped.add_render_callback(planner.render_control_solution)

    # reset environment
    poses = np.array(
        [
            [
                waypoints_track.raceline.xs[0],
                waypoints_track.raceline.ys[0],
                waypoints_track.raceline.yaws[0],
            ]
        ]
    )
    obs, info = env.reset(options={"poses": poses})
    done = False
    env.render()

    laptime = 0.0
    start = time.time()
    while not done:
        (steerv, accl), _ = planner.plan(obs["agent_0"])
        obs, timestep, terminated, truncated, infos = env.step(
            np.array([[steerv, accl]])
        )
        done = terminated or truncated
        laptime += timestep
        env.render()

        print(
            "speed: {}, steer vel: {}, accl: {}".format(
                obs["agent_0"]["linear_vel_x"], steerv, accl
            )
        )

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)


if __name__ == "__main__":
    main()
