#!/usr/bin/env python3
import logging

import numpy as np
import rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from std_msgs.msg import Int64, Bool


class PerceptionController(BaseHeadingController):
    def __init__(self):
        super().__init__()
        self.declare_parameter("active", True)
        self.image_detected =  False
        self.motor_sub = self.create_subscription(Bool, "/detector_bool",
                                                  self.detector_callback, 10)

    @property
    def active(self):
        return self.get_parameter("active").value

    def detector_callback(self, msg: Bool) -> None:
        """ sensor health callback triggered by subscription """
        if msg.data:
            self.image_detected = True

    def compute_control_with_goal(
        self, state: TurtleBotState, goal: TurtleBotState
    ) -> TurtleBotControl:

        ret = TurtleBotControl()
        ret.omega = 0.2 if not self.image_detected else 0.0
        return ret


if __name__ == "__main__":
    rclpy.init()
    node = PerceptionController()
    rclpy.spin(node)
    rclpy.shutdown()


