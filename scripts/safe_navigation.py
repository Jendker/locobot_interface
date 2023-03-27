#!/usr/bin/env python2
import time

import tf2_ros
from locobot_interface.srv import GetObstaclePosition
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import numpy as np
import rospy

from collections import namedtuple
Cylinder = namedtuple("Cylinder", "pt1 pt2 r")
Sphere = namedtuple("Sphere", "center r")
ClosestObstacle = namedtuple("ClosestObstacle", "distance angle")


class PointCloudMask:
    def __init__(self, tfBuffer, tfListener, rate):
        self.tfBuffer = tfBuffer
        self.tfListener = tfListener
        self.rate = rate

    def get_obstacle_points(self, points_to_check, pcl_time):
        cylinders = self._create_cylinders(pcl_time)
        sphere = self._create_sphere(pcl_time)
        discarded_indices = []
        for cylinder in cylinders:
            discarded_indices.append(self._points_in_cylinder(cylinder, points_to_check))
        discarded_indices.append(self._points_in_sphere(sphere, points_to_check))
        points_below = np.where(points_to_check[:, 2] < 0.05)
        discarded_indices.append(points_below)
        unique_discarded_indices = np.unique(np.concatenate(discarded_indices, axis=1))
        # revert mask
        mask = np.ones(points_to_check.shape[0], dtype=bool)
        mask[unique_discarded_indices] = False
        obstacle_points = points_to_check[mask, :]
        return obstacle_points

    def _create_sphere(self, time):
        sphere_frame = 'forearm_link'
        this_transform = self.tfBuffer.lookup_transform('base_link', sphere_frame, time).transform
        sphere_center = np.array(
            [this_transform.translation.x, this_transform.translation.y, this_transform.translation.z])
        radius = 0.1
        return Sphere(sphere_center, radius)

    def  _points_in_sphere(self, sphere, points_to_check):
        # Calculate the difference between the reference and measuring point
        diff = np.subtract(points_to_check, sphere.center)

        # Calculate square length of vector (distance between ref and point)^2
        dist = np.sum(np.power(diff, 2), axis=1)

        # If dist is less than radius^2, return True, else return False
        return np.where(dist < sphere.r ** 2)

    def _create_cylinders(self, time):
        frame_pairs = [['elbow_link', 'forearm_link'], ['forearm_link', 'wrist_link'], ['wrist_link', 'gripper_link'],
                       ['gripper_link', 'finger_r'], ['gripper_link', 'finger_l']]
        frame_positions = dict()
        for frame_pair in frame_pairs:
            for frame in frame_pair:
                if frame not in frame_positions:
                    try:
                        this_transform = self.tfBuffer.lookup_transform('base_link', frame, time).transform
                        this_point = np.array(
                            [this_transform.translation.x, this_transform.translation.y, this_transform.translation.z])
                        frame_positions[frame] = this_point
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        self.rate.sleep()
        radii = [0.1, 0.05, 0.05, 0.12, 0.12]
        cylinders = []
        for frame_pair, radius in zip(frame_pairs, radii):
            first_frame_point = frame_positions[frame_pair[0]]
            second_frame_point = frame_positions[frame_pair[1]]
            cylinders.append(Cylinder(first_frame_point, second_frame_point, radius))
        # add base cylinder
        zero_point = np.zeros(3)
        top_of_base = np.array([0., 0., 0.25])
        base_radius = 0.18
        cylinders.append(Cylinder(zero_point, top_of_base, base_radius))
        return cylinders

    def _points_in_cylinder(self, cylinder, points_to_check):
        vec = cylinder.pt2 - cylinder.pt1
        const = cylinder.r * np.linalg.norm(vec)
        return np.where((np.dot(points_to_check - cylinder.pt1, vec) >= 0) & (np.dot(points_to_check - cylinder.pt2, vec) <= 0) \
                        & (np.linalg.norm(np.cross(points_to_check - cylinder.pt1, vec), axis=1) <= const))


class SafeNavigation:
    def __init__(self):
        self.point_cloud_subscriber = rospy.Subscriber('/voxel_grid/pcl_downsampled', PointCloud2, self._pointcloud_callback)
        self.obstacle_pointcloud_pub = rospy.Publisher('obstacle_pcl', PointCloud2, queue_size=10)
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.service = rospy.Service('safe_navigation/get_obstacle_position', GetObstaclePosition, self._service_request)
        self.rate = rospy.Rate(5)
        self.position_requested = False
        self.last_closest_obstacle = None

    def _pointcloud_callback(self, message):
        if not self.position_requested:
            return
        self.position_requested = False
        self.last_closest_obstacle = self._calculate_distance(message)

    def _filter_points_with_mask(self, transformed_pointcloud):
        point_cloud_mask = PointCloudMask(self.tfBuffer, self.tfListener, self.rate)

        points = np.array(list(point_cloud2.read_points(transformed_pointcloud)))[:, :3]
        pcl_time = transformed_pointcloud.header.stamp
        obstacle_points = point_cloud_mask.get_obstacle_points(points, pcl_time)
        header = transformed_pointcloud.header
        # create pcl from points
        obstacle_pcl = point_cloud2.create_cloud_xyz32(header, obstacle_points)
        # TODO: remove this published after testing is finished
        self.obstacle_pointcloud_pub.publish(obstacle_pcl)
        return obstacle_points

    def _get_closest_obstacle_point_distance_angle(self, obstacle_points):
        distances = np.sqrt(np.sum(np.power(obstacle_points[:, :2], 2), axis=1))
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        angle = np.arctan2(obstacle_points[min_index][1], obstacle_points[min_index][0])
        return min_distance, angle

    def _calculate_distance(self, point_cloud_message):
        transform = self.tfBuffer.lookup_transform('base_link', 'camera_color_optical_frame', rospy.Time(0))
        transformed_pointcloud = do_transform_cloud(point_cloud_message, transform)
        obstacle_points = self._filter_points_with_mask(transformed_pointcloud)
        min_distance, angle = self._get_closest_obstacle_point_distance_angle(obstacle_points)
        return ClosestObstacle(min_distance, angle)

    def _service_request(self, req):
        self.last_closest_obstacle = None
        self.position_requested = True
        while self.last_closest_obstacle is None:
            time.sleep(0.01)
        return {'distance': self.last_closest_obstacle.distance, 'angle': self.last_closest_obstacle.angle}

    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('safe_navigation')
    safe_navigation = SafeNavigation()

    safe_navigation.run()

