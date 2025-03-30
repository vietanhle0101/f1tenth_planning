import numpy as np
import gymnasium as gym
from f1tenth_gym.envs import F110Env
import time
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
# Change your controller
from f1tenth_planning.control import Nonlinear_Kinemtic_MPC_Planner as RoboracerController
from f1tenth_planning.utils.utils import input_steering_speed_to_angle, input_acceleration_to_speed
from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.config.dynamics_config import fullscale_params

import rclpy
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import MarkerArray, Marker
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy,DurabilityPolicy, LivelinessPolicy
from rclpy.node import Node

class ControlRosWrapper(Node):
    def __init__(self):
        super().__init__("control_node")
        # Load track waypoints
        waypoints_track : Track = Track.from_raceline_file(
            os.path.join(os.path.dirname(__file__), "ros_map.csv"),
            delimiter=";",
            skip_rows=3,
        )
        # Make sure the yaw is between 0 and 2*pi
        waypoints_track.raceline.yaws = np.arctan2(
            np.sin(waypoints_track.raceline.yaws),
            np.cos(waypoints_track.raceline.yaws),
        )

        # Create planner
        self.planner = RoboracerController(track=waypoints_track, params=fullscale_params())

        # ROS publishers and subscribers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        # Create a QoS profile with KeepLast policy and depth of 1
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, 
            durability=DurabilityPolicy.VOLATILE,
            liveliness=LivelinessPolicy.AUTOMATIC,
            depth=1)

        odom_topic = '/fixposition/odometry' # '/ego_racecar/odom' for f1tenth_gym, '/pf/pose/odom' for f1tenth car
        self.pose_sub = self.create_subscription(
            Odometry, odom_topic, self.pose_callback, qos_profile
        )
        
        self.delta = 0.0
        self.local_plan_pub = self.create_publisher(MarkerArray, "/local_plan", 10)
        self.mpc_solution_pub = self.create_publisher(MarkerArray, "/mpc_solution", 10)
        self.global_plan_pub = self.create_publisher(MarkerArray, "/global_plan", 10)

        self.global_plan_marker_array = self._init_marker_array(self.planner.waypoints.shape[0], color=(0.0, 0.0, 1.0))
        self.global_plan_marker_array = self._update_marker_array(self.global_plan_marker_array, self.planner.waypoints)

        self.local_plan_marker_array = self._init_marker_array(self.planner.config.N + 1, color=(0.0, 1.0, 0.0))
        self.mpc_solution_marker_array = self._init_marker_array(self.planner.config.N, color=(1.0, 0.0, 0.0))

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
    
    def _update_marker_array(self, marker_array : MarkerArray, points):
        num_update = len(points)
        if(len(marker_array.markers) < len(points)):
            num_update = len(marker_array.markers)
            
        for i in range(num_update):
            marker : Marker = marker_array.markers[i]
            marker.pose.position.x = float(points[i][0])
            marker.pose.position.y = float(points[i][1])
        for i in range(num_update, len(marker_array.markers)):
            marker = marker_array.markers[i]
            marker.action = Marker.DELETE
            
        return marker_array
    
    def publish_visualizations(self, local_plan, mpc_solution):
        self.local_plan_pub.publish(self._update_marker_array(self.local_plan_marker_array, local_plan))
        self.mpc_solution_pub.publish(self._update_marker_array(self.mpc_solution_marker_array, mpc_solution))
        self.global_plan_pub.publish(self.global_plan_marker_array)

    def pose_callback(self, pose_msg : Odometry):
        # Get pose
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y

        pose = pose_msg.pose.pose
        twist = pose_msg.twist.twist
        beta = np.arctan2(twist.linear.y, twist.linear.x)
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        euler = R.from_quat(quaternion).as_euler('xyz', degrees=False)
        theta = euler[2]  # Yaw is the third element

        state_dict = {
            "pose_x": x,
            "pose_y": y,
            "delta": self.delta,
            "linear_vel_x": twist.linear.x,
            "pose_theta": theta,
            "ang_vel_z": twist.angular.z,
            "beta": beta
        }

        # Plan control commands
        steer_v, accel = self.planner.plan(state_dict)

        self.publish_visualizations(np.array(self.planner.ref_traj[:2].T),
                                    np.array(self.planner.x_pred[:2, :].T))
        
        # Convert steer_v and accel to steering angle and speed
        steer = input_steering_speed_to_angle(self.delta, steer_v, self.planner.config.dt)
        speed = input_acceleration_to_speed(accel, twist.linear.x, self.planner.config.dt)

        self.delta = steer
        # Publish control commands
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    print("Starting control node...")
    control_node = ControlRosWrapper()
    rclpy.spin(control_node)

    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()