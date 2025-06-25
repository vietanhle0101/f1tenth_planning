import numpy as np
import gymnasium as gym
from f1tenth_gym.envs import F110Env
import time
import os
import sys
import argparse
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
# Change your controller
#from f1tenth_planning.control import (
#    Nonlinear_Dynamic_MPC_Planner as RoboracerController,
#)

#from f1tenth_planning.control import (
#    Nonlinear_Kinemtic_MPC_Planner as RoboracerController,
#)
from f1tenth_planning.control import (
   Nonlinear_Dynamic_MPPI_Planner as RoboracerController,
)

from f1tenth_planning.utils.utils import (
    input_steering_speed_to_angle,
    input_acceleration_to_speed,
)
from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.config.dynamics_config import (
    fullscale_params,
    f1fifth_params,
    update_config_from_dict,
)
from f1tenth_gym.envs.action import SteerActionEnum, LongitudinalActionEnum

import rclpy
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import MarkerArray, Marker
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    DurabilityPolicy,
    LivelinessPolicy,
)
from rclpy.node import Node

try:
    from context_msgs.msg import STControl, STState, STCombined
    from context_msgs.msg import ParamList

    GT_STATE_PUB = True
except ImportError:
    GT_STATE_PUB = False
    print("context_msgs not found, GT_STATE_PUB will not be used.")


class ControlRosWrapper(Node):
    def __init__(self, scaling_factor=1.0, raceline='trajectory_logs.csv'):
        super().__init__("control_node")
        # Load track waypoints
        waypoints_track: Track = Track.from_raceline_file(
            os.path.abspath(os.path.expanduser(f"~/new_ws/trajectory_logs/{raceline}")),
            # raceline, # If you want to use a relative path or full path
            # os.path.join(os.path.dirname(__file__), raceline), # Current directory
            delimiter=";",
            skip_rows=3,
        )

        # Multiply the velocity by a factor
        waypoints_track.raceline.vxs *= scaling_factor

        # Create planner
        self.planner = RoboracerController(
            track=waypoints_track, params=f1fifth_params()
        )
        self.params = f1fifth_params()

        # ROS publishers and subscribers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        # Create a QoS profile with KeepLast policy and depth of 1
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            liveliness=LivelinessPolicy.AUTOMATIC,
            depth=1,
        )

        if not GT_STATE_PUB:
            odom_topic = "/fixposition/odometry"  # '/ego_racecar/odom' for f1tenth_gym, '/pf/pose/odom' for f1tenth car
            self.pose_sub = self.create_subscription(
                Odometry, odom_topic, self.pose_callback, qos_profile
            )
        else:
            self.pose_sub = self.create_subscription(
                STCombined, "/ground_truth/combined", self.pose_callback, qos_profile
            )

        self.param_sub = self.create_subscription(
            ParamList, "/estimates/current", self.param_update_callback, 10
        )

        self.delta = 0.0
        self.local_plan_pub = self.create_publisher(
            MarkerArray, "/control/local_plan", 10
        )
        self.mpc_solution_pub = self.create_publisher(
            MarkerArray, "/control/mpc_solution", 10
        )
        self.global_plan_pub = self.create_publisher(
            MarkerArray, "/control/global_plan", 10
        )

        self.global_plan_marker_array = self._init_marker_array(
            self.planner.waypoints.shape[0], color=(0.0, 0.0, 1.0)
        )
        self.global_plan_marker_array = self._update_marker_array(
            self.global_plan_marker_array, self.planner.waypoints
        )

        self.local_plan_marker_array = self._init_marker_array(
            self.planner.solver.config.N + 1, color=(0.0, 1.0, 0.0)
        )
        self.mpc_solution_marker_array = self._init_marker_array(
            self.planner.solver.config.N, color=(1.0, 0.0, 0.0)
        )
        self.get_logger().info("Controller node initialized")

    def _init_marker_array(self, num_markers, color=(1.0, 0.0, 1.0)):
        marker_array = MarkerArray()
        for i in range(num_markers):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.id = i
            marker.scale.x = 1.0
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.2
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.color.a = 1.0
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker_array.markers.append(marker)
        return marker_array

    def _update_marker_array(self, marker_array: MarkerArray, points):
        num_update = len(points)
        if len(marker_array.markers) < len(points):
            num_update = len(marker_array.markers)

        for i in range(num_update):
            marker: Marker = marker_array.markers[i]
            marker.pose.position.x = float(points[i][0])
            marker.pose.position.y = float(points[i][1])
        for i in range(num_update, len(marker_array.markers)):
            marker = marker_array.markers[i]
            marker.action = Marker.DELETE

        return marker_array

    def publish_visualizations(self, local_plan, mpc_solution):
        # Check for subscribers before publishing
        if self.local_plan_pub.get_subscription_count() > 0:
            self.local_plan_pub.publish(
                self._update_marker_array(self.local_plan_marker_array, local_plan)
            )
        if self.mpc_solution_pub.get_subscription_count() > 0:
            self.mpc_solution_pub.publish(
                self._update_marker_array(self.mpc_solution_marker_array, mpc_solution)
            )
        if self.global_plan_pub.get_subscription_count() > 0:
            self.global_plan_pub.publish(self.global_plan_marker_array)

    def pose_callback(self, pose_msg: Odometry):
        if GT_STATE_PUB:
            self.delta = pose_msg.control.steering_angle
            state_dict = {
                "pose_x": pose_msg.state.x,
                "pose_y": pose_msg.state.y,
                "delta": self.delta,
                "linear_vel_x": pose_msg.state.velocity,
                "pose_theta": pose_msg.state.yaw,
                "ang_vel_z": pose_msg.state.yaw_rate,
                "beta": pose_msg.state.slip_angle,
            }
        else:
            # Get pose
            x = pose_msg.pose.pose.position.x
            y = pose_msg.pose.pose.position.y

            pose = pose_msg.pose.pose
            twist = pose_msg.twist.twist
            beta = np.arctan2(twist.linear.y, twist.linear.x)
            quaternion = [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
            euler = R.from_quat(quaternion).as_euler("xyz", degrees=False)
            theta = euler[2]  # Yaw is the third element

            state_dict = {
                "pose_x": x,
                "pose_y": y,
                "delta": self.delta,
                "linear_vel_x": twist.linear.x,
                "pose_theta": theta,
                "ang_vel_z": twist.angular.z,
                "beta": beta,
            }

        # print(f"State: {state_dict}")
        # Plan control commands
        action, info = self.planner.plan(state_dict, params=self.params)
        steer_action = float(action[0])
        longitudtinal_action = float(action[1])

        self.publish_visualizations(
            np.array(self.planner.ref_traj[:2].T),
            np.array(self.planner.x_pred[:2, :].T),
        )

        # Convert steer_action to steering angle if needed
        steer = float(info["steering_angle"])

        # Convert longitudinal_action to speed if needed
        speed = float(info["velocity"])

        self.delta = steer
        # Publish control commands
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle_velocity = steer_action
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.acceleration = longitudtinal_action
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)
        # self.get_logger().info(
        #     f"Steering angle: {steer:.2f}, Speed: {speed:.2f}, Delta: {self.delta:.2f}"
        # )

    def param_update_callback(self, param_msg: ParamList):
        # Create a dict from param_msg.Params
        param_dict = {}
        for param in param_msg.params:
            param_dict[param.name] = param.value
        # Update the planner parameters, will be updated internally if supported by controller
        update_config_from_dict(self.params, param_dict)


def main(args=None):
    parser = argparse.ArgumentParser(description="Control ROS Wrapper Node")
    parser.add_argument(
        "--vx-scaling",
        default=1.0,
        help="Scale the velocity of the waypoints by this factor", 
        type=float,
    )
    parser.add_argument(
        "--raceline",
        default="trajectory_logs.csv",
        help="Path to the raceline file",
        type=str,
    )
    parsed_args, _ = parser.parse_known_args()

    scaling_factor = parsed_args.vx_scaling
    raceline = parsed_args.raceline

    rclpy.init(args=args)
    print("Starting control node...")
    control_node = ControlRosWrapper(scaling_factor=scaling_factor, raceline=raceline)
    rclpy.spin(control_node)

    control_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
