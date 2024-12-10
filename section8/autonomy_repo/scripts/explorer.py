#!/usr/bin/env python3
from typing import Optional
import numpy as np
from scipy.signal import convolve2d
import random

import rclpy  # ROS Client Library for Python
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid

from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer_node")  # Initialize the node
        self.get_logger().info("Frontier Explorer Node Initialized!")

        self.state: Optional[TurtleBotState] = None
        self.occupancy: Optional[StochOccupancyGrid2D] = None
        self.next_state: Optional[TurtleBotState] = None

        # Subscriptions
        self.state_sub = self.create_subscription(
            TurtleBotState, "/state", self.state_callback, 10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10
        )
        self.detector_sub = self.create_subscription(
            Bool, "/detector_bool", self.detector_callback, 10
        )

        # Publisher
        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)

        # Timer for publishing new goals
        self.timer = self.create_timer(10.0, self.publish_new_goal)

    def state_callback(self, msg: TurtleBotState) -> None:
        """Callback for receiving the latest TurtleBot state."""
        self.state = msg

    def map_callback(self, msg: OccupancyGrid) -> None:
        """Callback triggered when the map is updated."""
        occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )

        if self.occupancy is None:
            # Publish an initial exploration state
            next_state = self.explore(occupancy)
            self.cmd_nav_pub.publish(next_state)

        self.occupancy = occupancy

    def detector_callback(self, msg: Bool) -> None:
        """Callback triggered by the detector."""
        if msg.data:
            self.get_logger().info("Object detected! Stopping exploration!")
            self.timer.cancel()
            self.cmd_nav_pub.publish(self.next_state)

    def publish_new_goal(self) -> None:
        """Publish a new goal for exploration."""
        if self.occupancy is not None:
            next_state = self.explore(self.occupancy)
            self.cmd_nav_pub.publish(next_state)
            self.next_state = next_state

    def explore(self, occupancy: StochOccupancyGrid2D) -> TurtleBotState:
        """
        Determines potential exploration states.
        Args:
            occupancy: StochOccupancyGrid2D object representing the environment.
        Returns:
            A TurtleBotState representing the next exploration goal.
        """
        window_size = 13  # Neighborhood window size
        center_idx = (window_size - 1) // 2

        # Define the convolutional window
        window = np.ones((window_size, window_size))
        window[center_idx, center_idx] = 0

        # Find unknown areas
        unknown_filter = convolve2d(occupancy.probs.T == -1, window, mode="same")
        unknown_ok = unknown_filter > ((window_size * window_size) - 1) * 0.2

        # Ensure no occupied cells in the window
        occupied_filter = convolve2d(occupancy.probs.T > 0, window, mode="same")
        occupied_ok = occupied_filter == 0

        # Favor unoccupied areas
        unoccupied_filter = convolve2d(occupancy.probs.T == 0, window, mode="same")
        unoccupied_ok = unoccupied_filter > ((window_size * window_size) - 1) * 0.3

        # Identify frontier cells
        frontier = np.argwhere(unknown_ok & occupied_ok & unoccupied_ok)
        frontier_states = occupancy.grid2state(frontier)

        if frontier.size == 0:
            self.get_logger().info("Frontier is empty! Stopping exploration!")
            self.timer.cancel()
            return self.next_state

        # Select the next state
        dists = np.linalg.norm(
            frontier_states - np.array([self.state.x, self.state.y]), axis=1
        )
        closest_idx = np.argmin(dists)
        next_state = TurtleBotState(
            x=frontier_states[closest_idx, 0],
            y=frontier_states[closest_idx, 1],
        )
        return next_state


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)  # Keep the node running
    rclpy.shutdown()


if __name__ == "__main__":
    main()
