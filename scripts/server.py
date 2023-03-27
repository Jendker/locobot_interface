#!/usr/bin/python

import traceback
import rospy

from pyrobot import Robot
from locobot_interface.srv import *
from kobuki_msgs.msg import SensorState

from sensor_msgs.msg import Image

import time
import numpy as np


class LocobotServer:
    """ Server for remote PyRobot usage.
    Wraps the PyRobot API for the LoCoBot base/arm/camera.
    """

    def __init__(self):
        self.bot = Robot('locobot')

        self.bot.base.configs['BASE']['MAX_ABS_FWD_SPEED'] = 3.
        self.bot.base.configs['BASE']['MAX_ABS_TURN_SPEED'] = 6.

        self.arm_joint_cmd_srv = rospy.Service('python3_server/arm_joint_command', ArmJointCommand,
                                               self.arm_joint_cmd_callback)
        self.arm_ee_cmd_srv = rospy.Service('python3_server/arm_ee_command', ArmEECommand, self.arm_ee_cmd_callback)
        self.gripper_cmd_srv = rospy.Service('python3_server/gripper_command', GripperCommand,
                                             self.gripper_cmd_callback)

        self.base_pos_cmd_srv = rospy.Service('python3_server/base_pos_command', BasePositionCommand,
                                              self.base_pos_cmd_callback)
        self.base_vel_cmd_srv = rospy.Service('python3_server/base_vel_command', BaseVelocityCommand,
                                              self.base_vel_cmd_callback)

        self.pointcloud_srv = rospy.Service('python3_server/get_pointcloud', GetPointcloud, self.pointcloud_callback)
        self.fk_srv = rospy.Service('python3_server/get_fk', GetFK, self.get_fk_callback)
        self.ik_srv = rospy.Service('python3_server/get_ik', GetIK, self.get_ik_callback)
        self.pan_tilt_srv = rospy.Service('python3_server/pan_tilt', SetPanTilt, self.pan_tilt_callback)

        self.color_srv = rospy.Service('python3_server/color', GetImage, self.color_callback)
        self.depth_srv = rospy.Service('python3_server/depth', GetImage, self.depth_callback)

        self.grasp_obstructed_srv = rospy.Service('python3_server/get_grasp_obstructed', GetGraspObstructed,
                                                  self.grasp_obstructed_callback)

        # self.obstacle_srv = rospy.ServiceProxy('/safe_navigation/get_obstacle_position', GetObstaclePosition)

        # self.sensors_sub = rospy.Subscriber('mobile_base/sensors/core', SensorState, self.sensors_callback)

        self.last_sensors_reading = None
        self.stuck_occurances = 0
        self.stuck_occurances_limit = 1

        self.arm_link_names = ["arm_base_link", "shoulder_link", "elbow_link", "forearm_link", "wrist_link",
                               "gripper_link"]

    def print_joint_pos(self):
        print("print_joint_pos:")
        joint_positions = self.bot.arm.get_joint_angles()
        for name in self.arm_link_names:
            pos, rot = self.bot.arm.compute_fk_position(joint_positions, name)
            print(name, pos.squeeze())

    def is_joint_positions_obstructed_link(self, joint_positions, link_name, pts, radius):
        link_pos, _ = self.bot.arm.compute_fk_position(joint_positions, link_name)
        link_pos = link_pos.squeeze()

        dists = np.sum(np.square(pts - link_pos), axis=1)

        num_obstructed = np.sum(dists <= radius * radius)

        return num_obstructed >= 2

    def is_joint_positions_obstructed(self, positions):
        pts, _ = self.bot.camera.get_current_pcd(in_cam=False)
        z_mask = pts[:, 2] >= 0.08
        pts = pts[z_mask]

        return (
                self.is_joint_positions_obstructed_link(positions, self.arm_link_names[2], pts, 0.04) or
                self.is_joint_positions_obstructed_link(positions, self.arm_link_names[3], pts, 0.04) or
                self.is_joint_positions_obstructed_link(positions, self.arm_link_names[4], pts, 0.05) or
                self.is_joint_positions_obstructed_link(positions, self.arm_link_names[5], pts, 0.05)
        )

    def arm_joint_cmd_callback(self, req):
        print("arm_joint_cmd_callback")
        try:
            positions = [req.joint_1, req.joint_2, req.joint_3, req.joint_4, req.joint_5]

            success = False
            if req.check_obstruction and self.is_joint_positions_obstructed(positions):
                rospy.logerr("desired joint positions obstructed")
            else:
                success = self.bot.arm.set_joint_positions(positions, plan=req.plan, wait=req.wait)
            if req.wait:
                return success
            else:
                return True
        except Exception as e:
            traceback.print_exc(e)
            return False

    def arm_ee_cmd_callback(self, req):
        print("arm_ee_cmd_callback")
        try:
            position = [req.x, req.y, req.z]
            pitch = req.pitch
            roll = req.roll
            plan = req.plan
            wait = req.wait

            position = np.array(position).flatten()
            base_offset, _, _ = self.bot.arm.get_transform(
                self.bot.arm.configs.ARM.ARM_BASE_FRAME, "arm_base_link"
            )
            yaw = np.arctan2(position[1] - base_offset[1], position[0] - base_offset[0])

            if roll is None:
                # read the current roll angle
                roll = -self.bot.arm.get_joint_angle("joint_5")
            euler = np.array([yaw, pitch, roll], dtype=np.float64)

            joint_positions = self.bot.arm.compute_ik(position, euler, numerical=True)
            success = False

            if joint_positions is None:
                rospy.logerr("No IK solution found; check if target_pose is valid")
            elif req.check_obstruction and self.is_joint_positions_obstructed(joint_positions):
                rospy.logerr("desired end effector positions obstructed")
            else:
                success = self.bot.arm.set_joint_positions(joint_positions, plan=plan, wait=wait)

            # success = self.bot.arm.set_ee_pose_pitch_roll(positions, req.pitch, roll=req.roll,
            #                                              plan=req.plan, wait=req.wait)
            if wait:
                return success
            else:
                return True
        except Exception as e:
            traceback.print_exc(e)
            return False

    def gripper_cmd_callback(self, req):
        print("gripper_cmd_callback")
        try:
            bot_wait = False
            if req.command == 'open':
                self.bot.gripper.open(bot_wait)
            elif req.command == 'close':
                self.bot.gripper.close(bot_wait)
            elif req.command == 'force_close_if_open':
                self.bot.gripper.force_close_if_open(bot_wait)
            else:
                raise ValueError('Unknown gripper command')
            if req.wait:
                time.sleep(1)
            print("done")
            return True
        except Exception as e:
            traceback.print_exc(e)
            return False

    def base_pos_cmd_callback(self, req):
        try:
            position = [req.x, req.y, req.t]
            method = self.bot.base.go_to_relative if req.relative else self.bot.base.go_to_absolute
            success = method(position, close_loop=req.close_loop, smooth=req.smooth)
            return success
        except Exception as e:
            traceback.print_exc(e)
            return False

    def get_smallest_encoder_difference(self, previous_sensors_reading):
        left_enc_difference = min(
            [np.abs(previous_sensors_reading.left_encoder - self.last_sensors_reading.left_encoder),
             np.abs(previous_sensors_reading.left_encoder - self.last_sensors_reading.left_encoder - 2 ^ 16)])
        right_enc_difference = min(
            [np.abs(previous_sensors_reading.right_encoder - self.last_sensors_reading.right_encoder),
             np.abs(previous_sensors_reading.right_encoder - self.last_sensors_reading.right_encoder - 2 ^ 16)])
        return min([left_enc_difference, right_enc_difference])

    def is_robot_stuck(self, req, previous_sensors_reading):
        smallest_encoder_difference = self.get_smallest_encoder_difference(previous_sensors_reading)
        if np.abs(req.fwd_speed) >= 0.1:
            if smallest_encoder_difference < 1000:
                self.stuck_occurances += 1
            else:
                self.stuck_occurances = 0
        elif np.abs(req.turn_speed) >= 0.5:
            if smallest_encoder_difference < 300:
                self.stuck_occurances += 1
            else:
                self.stuck_occurances = 0
        return self.stuck_occurances > self.stuck_occurances_limit

    def base_vel_cmd_callback(self, req):
        print("base_vel_cmd_callback")
        try:
            # previous_sensor_reading = self.last_sensors_reading
            success = self.bot.base.set_vel(req.fwd_speed, req.turn_speed, req.exe_time)

            # rospy.sleep(0.5)
            # if self.bot.base.base_state.bumper:
            #     print("bumper")
            #     self.bot.base.set_vel(-0.1, 0, 1.0)
            #     self.bot.base.base_state.bumper = False

            # check if is stuck
            # if self.is_robot_stuck(req, previous_sensor_reading):
            #    error_text = 'Robot is stuck. Shutting down.'
            #    rospy.logerr(error_text)
            #    rospy.signal_shutdown(error_text)

            turn_dir = self.get_turn_dir(req.more)
            if turn_dir is not None:
                print("turn", turn_dir)
                # self.bot.base.set_vel(-0.1, 0, 0.5)
                for i in range(5):
                    print(i)
                    if turn_dir == "left":
                        self.bot.base.set_vel(0, 1.0, 1)
                    else:
                        self.bot.base.set_vel(0, -1.0, 1)
                    rospy.sleep(0.25)
                    # rospy.sleep(1.0)
                    done = self.get_turn_dir(req.more) is None
                    if done:
                        break
            # rospy.sleep(0.2)
            print("done")
            return True
        except Exception as e:
            traceback.print_exc(e)
            return False

    def color_callback(self, req):
        print("color_callback")
        try:
            image = self.bot.camera.get_rgb()
            image_msg = self.bot.camera.cv_bridge.cv2_to_imgmsg(image)
            print("done")
            return image_msg
        except Exception as e:
            traceback.print_exc(e)
            return None

    def depth_callback(self, req):
        print("color_callback")
        try:
            image = self.bot.camera.get_depth()*1000
            image = image.astype(np.uint16)
            image_msg = self.bot.camera.cv_bridge.cv2_to_imgmsg(image, "16UC1")
            print("done")
            return image_msg
        except Exception as e:
            traceback.print_exc(e)
            return None

    def get_fk_callback(self, req):
        return self.bot.compute_fk_position(req.joint_positions, 'base_link')

    def get_ik_callback(self, req):
        position = np.array(req.position).flatten()
        base_offset, _, _ = self.bot.arm.get_transform(
            self.bot.arm.configs.ARM.ARM_BASE_FRAME, "arm_base_link"
        )
        yaw = np.arctan2(position[1] - base_offset[1], position[0] - base_offset[0])
        euler = np.array([yaw, req.pitch, req.roll], dtype=np.float64)
        joint_positions = self.bot.arm.compute_ik(position, euler, numerical=True)
        return [joint_positions]

    def get_turn_dir(self, more):
        pts, _ = self.bot.camera.get_current_pcd(in_cam=False)

        # remove ground
        z_mask = (pts[:, 2] >= 0.08)
        pts = pts[z_mask]

        # remove robot body
        r2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
        r_mask = (r2 >= 0.18 ** 2)
        pts = pts[r_mask]
        r2 = r2[r_mask]

        # remove grippper
        wrist_link_pos = np.array([0.168, 0.0, 0.215])
        gripper_link_pos = np.array([0.237, 0.0, 0.120])

        wrist_mask = np.sum(np.square(pts - wrist_link_pos), axis=1) >= 0.13 ** 2
        gripper_mask = np.sum(np.square(pts - gripper_link_pos), axis=1) >= 0.13 ** 2
        wrist_gripper_mask = np.logical_and(wrist_mask, gripper_mask)

        pts = pts[wrist_gripper_mask]
        r2 = r2[wrist_gripper_mask]

        # split into left and right
        left_mask = pts[:, 1] >= 0.0
        right_mask = pts[:, 1] < 0.0

        left_r2 = r2[left_mask]
        right_r2 = r2[right_mask]

        # compare min points
        min_left_r2 = np.min(left_r2, initial=1000.0)
        min_right_r2 = np.min(right_r2, initial=1000.0)

        if more:
            thresh = 0.4
        else:
            thresh = 0.36

        if min_left_r2 < min_right_r2:
            if min_left_r2 <= thresh ** 2:
                return "right"
        else:
            if min_right_r2 <= thresh ** 2:
                return "left"
        return None

    # def filter_pc_grasp(self, x, y, pts):
    #     if y >= 0:
    #         y_min = -0.063
    #         y_max = y + 0.045
    #     else:
    #         y_max = 0.045
    #         y_min = y - 0.063

    #     x_min = 0.277
    #     x_max = max(0.41, x + 0.06)

    #     z_min = 0.09

    #     x_mask = np.logical_and(x_min <= pts[:, 0], pts[:, 0] <= x_max)
    #     y_mask = np.logical_and(y_min <= pts[:, 1], pts[:, 1] <= y_max)
    #     z_mask = pts[:, 2] >= z_min

    #     return pts[np.logical_and(np.logical_and(x_mask, y_mask), z_mask)]

    def grasp_obstructed_callback(self, req):
        print("grasp_obstructed_callback")
        # if self.old:
        #    return False

        threshold = 10
        try:
            position = [req.x, req.y, 0.20]
            pitch = np.pi / 2.0
            roll = 0.0

            position = np.array(position).flatten()
            base_offset, _, _ = self.bot.arm.get_transform(
                self.bot.arm.configs.ARM.ARM_BASE_FRAME, "arm_base_link"
            )
            yaw = np.arctan2(position[1] - base_offset[1], position[0] - base_offset[0])

            euler = np.array([yaw, pitch, roll], dtype=np.float64)

            joint_positions = self.bot.arm.compute_ik(position, euler, numerical=True)

            return self.is_joint_positions_obstructed(joint_positions)

            # pts, _ = self.bot.camera.get_current_pcd(in_cam=False)
            # filtered_grasps = self.filter_pc_grasp(x, y, pts)
            # print("done")
            # return filtered_grasps.shape[0] >= threshold
        except Exception as e:
            traceback.print_exc(e)
            return True

    def pointcloud_callback(self, req):
        print("pointcloud_callback")
        try:
            if req.in_cam:
                pts, colors = self.bot.camera.get_current_pcd(True)
            else:
                pts, colors = self.bot.camera.get_current_pcd(False)
                if req.filter_pts:
                    mask = pts[:, 2] > .04
                    pts = pts[mask]
            return [pts.reshape(-1).tolist()]
        except Exception as e:
            traceback.print_exc(e)
            return []

    def pan_tilt_callback(self, req):
        print("pan_tilt_callback")
        try:
            self.bot.camera.set_pan(req.pan)
            self.bot.camera.set_tilt(req.tilt)
            print("done")
            return True
        except Exception as e:
            traceback.print_exc(e)
            return False

    def sensors_callback(self, req):
        self.last_sensors_reading = req


if __name__ == '__main__':
    s = LocobotServer()
    print('Ready for service')
    rospy.spin()
