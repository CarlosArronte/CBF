#!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node

from std_msgs.msg import Float32
from nav_msgs.msg import Odometry

# import serial
import numpy as np


class MLP2NO:
    def __init__(self):
        
        # =========================
        # Vehicle parameters
        # =========================
        self.L = 0.33                        # wheelbase [m]
        self.delta_max = np.pi / 9.0         # 20 deg

        # =========================
        # Acceleration limits
        # =========================
        self.a_min = -3.0
        self.a_max =  3.0

        # =========================
        # RC parameters
        # =========================
        self.rc_min = 1000.0
        self.rc_max = 2000.0
        self.rc_neutral = 1500.0
        self.radio_tolerance = 50.0

        # =========================
        # States
        # =========================
        self.v = 0.0
        self.accel = 0.0
        self.delta = 0.0
    

    # =========================
    # Utility
    # =========================
    def map_value(self, x, in_min, in_max, out_min, out_max):
        x = np.clip(x, in_min, in_max)
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def filter_mlp(self, mlp_params):
        
        self.v = float(mlp_params['odometry_speed'])       # speed from odometry [m/s]
        trottle = float(mlp_params['throttle'])   # throttle (RC or physical accel)
        steering = float(mlp_params['steering'])  # steering (RC or steering angle)

        self.accel = self.map_value(
            trottle,
            self.rc_min, self.rc_max,
            self.a_min, self.a_max
        )

        self.accel = np.clip(self.accel, self.a_min, self.a_max)

         
        self.delta = self.map_value(
            steering,
            self.rc_min, self.rc_max,
            -self.delta_max, self.delta_max
        )

        self.delta = np.clip(self.delta, -self.delta_max, self.delta_max)
           

        return [self.delta,self.accel]

