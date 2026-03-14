#!/usr/bin/env python3



from std_msgs.msg import Float32, Float64
from nav_msgs.msg import Odometry

import numpy as np


class CBF2Vehicle():

    def __init__(self):
        super().__init__('cbf2vehicle')

        # Publishers
        self.throttle_pub = self.create_publisher(
            Float64, '/commands/motor/speed', 10
        )
        self.steering_pub = self.create_publisher(
            Float64, '/commands/servo/position', 10
        )

        # Subscribers
        self.create_subscription(
            Float32,
            '/autodrive/f1tenth_1/steering_command',
            self.steering_callback,
            10
        )
        self.create_subscription(
            Float32,
            '/autodrive/f1tenth_1/throttle_command',
            self.throttle_callback,
            10
        )
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # =========================
        # Internal state
        # =========================
        self.acc_cmd = 0.0
        self.steering_cmd = 0.0

        self.v_cmd = 0.0      # integrated commanded speed [m/s]
        self.v_odom = 0.0     # measured speed [m/s]

        self.last_acc_time = self.get_clock().now()

        # =========================
        # Parameters
        # =========================
        self.dt = 0.025            # 40 Hz
        self.max_speed = 5.0      # [m/s]

        self.motor_min = 1500.0
        self.motor_max = 8000.0 #4000.0

        self.servo_min = 0.15
        self.servo_max = 0.85

        # Invert motor rotation direction
        self.invert_motor_rotation_direction = -1

        # Optional safety: after this time, accel → 0
        self.acc_timeout = 0.5  # [s]

        self.timer = self.create_timer(self.dt, self.timer_callback)

    def steering_rad_to_servo(self,steering_cmd):
        # Convert radians → degrees
        steering_deg = np.degrees(steering_cmd)

        # Optional: clamp to valid range
        steering_deg = np.clip(steering_deg, -20.0, 20.0)

        # Linear mapping
        servo_cmd = 0.15 + (steering_deg + 20.0) * (0.85 - 0.15) / (40.0)

        return servo_cmd


    # =====================================================
    # Callbacks
    # =====================================================
    def steering_callback(self, msg):
        self.steering_cmd = msg.data

    def throttle_callback(self, msg):
        self.acc_cmd = max(min(msg.data, 3.0), -3.0)
        self.last_acc_time = self.get_clock().now()

    def odom_callback(self, msg):
        self.v_odom = msg.twist.twist.linear.x

    # =====================================================
    # Main loop
    # =====================================================
    def timer_callback(self):

        # ---------------------------------
        # Acceleration timeout (optional)
        # ---------------------------------
        time_since_acc = (
            self.get_clock().now() - self.last_acc_time
        ).nanoseconds * 1e-9

        if time_since_acc > self.acc_timeout:
            self.acc_cmd = 0.0   # coast

        # ---------------------------------
        # Integrate acceleration (COASTING OK)
        # ---------------------------------
        self.v_cmd += self.acc_cmd * self.dt

        # Enforce limits
        if self.v_cmd < 0.0:
            self.v_cmd = 0.0
        if self.v_cmd > self.max_speed:
            self.v_cmd = self.max_speed

        # ---------------------------------
        # Speed → Motor command
        # ---------------------------------
        if self.v_cmd > 0.0:
            motor_cmd = (
                self.motor_min +
                (self.motor_max - self.motor_min)
                * (self.v_cmd / self.max_speed)
            )
        else:
            motor_cmd = 0.0

        motor_cmd = max(0.0, min(motor_cmd, self.motor_max))

        throttle_msg = Float64()
        throttle_msg.data = (
            self.invert_motor_rotation_direction * motor_cmd
        )
        self.throttle_pub.publish(throttle_msg)

        # ---------------------------------
        # Steering (published continuously)
        # ---------------------------------
        steering_msg = Float64()
        steering_msg.data = self.steering_rad_to_servo(self.steering_cmd)

        self.steering_pub.publish(steering_msg)

# def main(args=None):
#     rclpy.init(args=args)
#     node = CBF2Vehicle()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()
