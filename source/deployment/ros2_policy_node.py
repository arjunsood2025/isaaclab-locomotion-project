"""ROS2 inference node for a trained (blind) locomotion policy.

Runs the exported TorchScript policy at a fixed control rate, reproducing the exact
observation layout used in training (see ObservationsCfg.PolicyCfg):

    [ base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) | command(3) |
      joint_pos_rel(12) | joint_vel(12) | last_action(12) ]                 = 48 dims

Subscribes:
    /joint_states  (sensor_msgs/JointState)   joint positions + velocities
    /imu           (sensor_msgs/Imu)          orientation + angular velocity
    /odom          (nav_msgs/Odometry)        base linear velocity estimate*
    /cmd_vel       (geometry_msgs/Twist)      operator velocity command

Publishes:
    /policy/joint_position_targets (std_msgs/Float64MultiArray), 12 targets in the
    policy's joint order, to be consumed by the robot's low-level PD controller
    (kp=25, kd=0.5 to match training).

* Real robots have no ground-truth base linear velocity. Options, in order of rigor:
  (a) run a state estimator (leg odometry + IMU EKF) and feed its output here;
  (b) retrain with base_lin_vel removed from the actor observation (1-line config
      change) -- common practice for real deployments;
  (c) feed zeros and accept degraded tracking (works surprisingly often because the
      policy leans mostly on joint state + IMU). The node warns if /odom is silent.

Safety features (all deliberately boring and explicit):
    - EMA action filter to remove residual high-frequency content
    - joint target clamping to soft limits
    - command watchdog: /cmd_vel silence => command decays to zero
    - state watchdog: stale /joint_states or /imu => hold last targets and warn
    - inference latency is measured and logged every second

Run:
    ros2 run <your_package> ros2_policy_node --ros-args \
        -p model_path:=exported/policy.pt -p control_rate_hz:=50.0
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

# Joint order the policy was trained with (Isaac Lab's Go2 articulation order).
# The incoming /joint_states message may order joints differently; we re-index by name.
POLICY_JOINT_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]

# Default standing pose in the same order (matches UNITREE_GO2_CFG init_state)
DEFAULT_JOINT_POS = np.array(
    [0.1, -0.1, 0.1, -0.1,  0.8, 0.8, 1.0, 1.0,  -1.5, -1.5, -1.5, -1.5],
    dtype=np.float32,
)

ACTION_SCALE = 0.25  # must match ActionsCfg.joint_pos.scale
NUM_JOINTS = 12
OBS_DIM = 48


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector v into the frame given by quaternion q (w, x, y, z)."""
    w, x, y, z = q
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w**2 - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


class PolicyNode(Node):
    def __init__(self):
        super().__init__("locomotion_policy_node")

        self.declare_parameter("model_path", "exported/policy.pt")
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("ema_alpha", 0.8)  # 1.0 = no smoothing
        self.declare_parameter("cmd_timeout_s", 0.5)
        self.declare_parameter("state_timeout_s", 0.1)
        self.declare_parameter("max_joint_delta_rad", 0.6)  # clamp around default pose

        model_path = self.get_parameter("model_path").value
        self.policy = torch.jit.load(model_path, map_location="cpu")
        self.policy.eval()
        self.get_logger().info(f"Loaded policy: {model_path}")

        # -- state buffers (filled by callbacks)
        self.joint_pos = DEFAULT_JOINT_POS.copy()
        self.joint_vel = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w, x, y, z
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)  # vx, vy, yaw_rate
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.filtered_targets = DEFAULT_JOINT_POS.copy()

        self.t_last_cmd = 0.0
        self.t_last_joint_state = 0.0
        self.t_last_imu = 0.0
        self.odom_seen = False
        self.latency_buffer: list[float] = []

        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, "/joint_states", self.on_joint_state, sensor_qos)
        self.create_subscription(Imu, "/imu", self.on_imu, sensor_qos)
        self.create_subscription(Odometry, "/odom", self.on_odom, sensor_qos)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)
        self.target_pub = self.create_publisher(
            Float64MultiArray, "/policy/joint_position_targets", 10
        )

        rate = self.get_parameter("control_rate_hz").value
        self.dt = 1.0 / rate
        self.create_timer(self.dt, self.control_step)
        self.create_timer(1.0, self.report_latency)
        self.get_logger().info(f"Control loop at {rate:.0f} Hz (budget {self.dt*1000:.1f} ms)")

    # ------------------------------------------------------------------ callbacks

    def on_joint_state(self, msg: JointState):
        # re-index from message order to policy order by joint name
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        for j, name in enumerate(POLICY_JOINT_ORDER):
            i = name_to_idx.get(name)
            if i is not None:
                self.joint_pos[j] = msg.position[i]
                if i < len(msg.velocity):
                    self.joint_vel[j] = msg.velocity[i]
        self.t_last_joint_state = time.monotonic()

    def on_imu(self, msg: Imu):
        self.base_quat = np.array(
            [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z],
            dtype=np.float32,
        )
        self.base_ang_vel = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            dtype=np.float32,
        )
        self.t_last_imu = time.monotonic()

    def on_odom(self, msg: Odometry):
        # odometry twist is body-frame per REP-105 when child_frame_id = base link
        self.base_lin_vel = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
            dtype=np.float32,
        )
        self.odom_seen = True

    def on_cmd_vel(self, msg: Twist):
        self.command = np.array(
            [msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32
        )
        self.t_last_cmd = time.monotonic()

    # ------------------------------------------------------------------ control

    def build_observation(self) -> np.ndarray:
        gravity_body = quat_rotate_inverse(
            self.base_quat, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )
        return np.concatenate([
            self.base_lin_vel,
            self.base_ang_vel,
            gravity_body,
            self.command,
            self.joint_pos - DEFAULT_JOINT_POS,
            self.joint_vel,
            self.last_action,
        ]).astype(np.float32)

    def control_step(self):
        now = time.monotonic()

        # state watchdog: stale sensors -> hold last safe targets, do not run the policy
        state_timeout = self.get_parameter("state_timeout_s").value
        if (now - self.t_last_joint_state > state_timeout
                or now - self.t_last_imu > state_timeout):
            self.get_logger().warn("Stale robot state -- holding last targets",
                                   throttle_duration_sec=1.0)
            self.publish_targets(self.filtered_targets)
            return

        # command watchdog: operator silence -> decay command to zero (graceful stop)
        if now - self.t_last_cmd > self.get_parameter("cmd_timeout_s").value:
            self.command *= 0.9

        if not self.odom_seen:
            self.get_logger().warn(
                "/odom silent: base_lin_vel = 0. See module docstring for options.",
                throttle_duration_sec=5.0,
            )

        t0 = time.perf_counter()
        obs = torch.from_numpy(self.build_observation()).unsqueeze(0)
        with torch.inference_mode():
            action = self.policy(obs).squeeze(0).numpy().astype(np.float32)
        self.latency_buffer.append((time.perf_counter() - t0) * 1000.0)

        self.last_action = action

        # action -> joint targets, exactly as in training
        targets = DEFAULT_JOINT_POS + ACTION_SCALE * action
        # safety clamp: never command further than max_delta from the default pose
        max_delta = self.get_parameter("max_joint_delta_rad").value
        targets = np.clip(targets, DEFAULT_JOINT_POS - max_delta, DEFAULT_JOINT_POS + max_delta)
        # EMA filter removes residual high-frequency content before the PD loop
        alpha = self.get_parameter("ema_alpha").value
        self.filtered_targets = alpha * targets + (1.0 - alpha) * self.filtered_targets

        self.publish_targets(self.filtered_targets)

    def publish_targets(self, targets: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in targets]
        self.target_pub.publish(msg)

    def report_latency(self):
        if not self.latency_buffer:
            return
        buf = sorted(self.latency_buffer)
        mean = sum(buf) / len(buf)
        p95 = buf[min(int(0.95 * len(buf)), len(buf) - 1)]
        budget = self.dt * 1000.0
        self.get_logger().info(
            f"inference latency: mean {mean:.2f} ms, p95 {p95:.2f} ms "
            f"({100 * p95 / budget:.0f}% of {budget:.0f} ms budget)"
        )
        self.latency_buffer.clear()


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
