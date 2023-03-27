#!/usr/bin/python

import traceback
import numpy as np
import rospy
from std_msgs.msg import Int8, Float64
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from kobuki_msgs.msg import BumperEvent, WheelDropEvent, PowerSystemEvent, Sound
from locobot_interface.srv import *


DEFAULT_PAN = 0.00153398083057
DEFAULT_TILT = 0.809941887856
MIN_PAN = -2.7
MAX_PAN = 2.6
MIN_TILT = -np.radians(80)
MAX_TILT = np.radians(100)
RESET_PAN = 0.0
RESET_TILT = 0.0
BB_SIZE = 5
JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
GRIPPER_JOINT_NAME = 'joint_7'

class LocobotClient:
    """ Client interface for remote PyRobot server. 
    Wraps the PyRobot API for the LoCoBot base/arm/camera.
    """
    def __init__(self, debug=False, pause_filepath=None):
        rospy.init_node('locobot_interface')

        self.arm_joint_cmd_client = rospy.ServiceProxy('/python3_server/arm_joint_command', ArmJointCommand)
        self.arm_ee_cmd_client = rospy.ServiceProxy('/python3_server/arm_ee_command', ArmEECommand)
        self.gripper_cmd_client = rospy.ServiceProxy('/python3_server/gripper_command', GripperCommand)

        self.joints_subscriber = rospy.Subscriber('/joint_states', JointState, self._joints_callback)
        self.gripper_subscriber = rospy.Subscriber('/gripper/state', Int8, self._gripper_callback)

        self.base_pos_cmd_client = rospy.ServiceProxy('/python3_server/base_pos_command', BasePositionCommand)
        self.base_vel_cmd_client = rospy.ServiceProxy('/python3_server/base_vel_command', BaseVelocityCommand)

        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self._odom_callback)
        self._odom_state = None

        self.bumper_subscriber = rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, self._bumper_callback)
        self.power_subscriber = rospy.Subscriber('/mobile_base/events/power_system', PowerSystemEvent, self._power_callback)
        self.drop_subscriber = rospy.Subscriber('/mobile_base/events/wheel_drop', WheelDropEvent, self._drop_callback)
        
        self.sound_publisher = rospy.Publisher('/mobile_base/commands/sound', Sound)

        self._bumper_state = 0
        self._power_state = 0
        self._drop_state = 0

        self._debug = debug
        self._pause_filepath = pause_filepath

        self.pointcloud_client = rospy.ServiceProxy('/python3_server/get_pointcloud', GetPointcloud)
        self.fk_client = rospy.ServiceProxy('python3_server/get_fk', GetFK)
        self.ik_client = rospy.ServiceProxy('python3_server/get_ik', GetIK)
        self.pan_tilt_client = rospy.ServiceProxy('/python3_server/pan_tilt', SetPanTilt)

        self.color_client = rospy.ServiceProxy('/python3_server/color', GetImage)
        self.depth_client = rospy.ServiceProxy('python3_server/depth', GetImage)

        self.grasp_obstructed_client = rospy.ServiceProxy('/python3_server/get_grasp_obstructed', GetGraspObstructed)

        # self.color_subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self._image_callback, queue_size=1)
        # self.depth_subscriber = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self._depth_callback, queue_size=1)

        rospy.sleep(1)

    def debug_print(self, *args, **kwargs):
        if self._debug:
            print(*args, **kwargs)

    def set_base_pos(self, x, y, t, relative=True, close_loop=False, smooth=False):
        """ Set position of the base relative to base frame or current position

        :param x,y: displacement/position along x-y axes
        :param t: goal base orientation (in radians)
        :param relative: interprets x,y relative to current position if True,
                            else interpreted as absolute coordinates
        :param close_loop: uses odometry during execution
        :param smooth: tries to smooth motion to goal

        :returns: True if successful; False otherwise
        """
        try:
            resp = self.base_pos_cmd_client(x, y, t, relative, close_loop, smooth)
            return resp.success
        except rospy.ServiceException as e:
            traceback.print_exc(e)
            return False

    def set_base_vel(self, fwd_speed, turn_speed, exe_time=1, more=False):
        """ Set velocity of the base

        :param left, right: desired velocity of the left/right wheel

        :returns: True if successful; False otherwise
        """
        try:
            self.debug_print("set_base_vel")
            resp = self.base_vel_cmd_client(fwd_speed, turn_speed, exe_time, more)
            self.debug_print("done")
            return resp.success
        except rospy.ServiceException as e:
            traceback.print_exc(e)
            return False

    def set_joint_angles(self, joints, plan=False, wait=True, check_obstruction=False):
        """ Set desired joint angles

        :param joints: list of desired joint angles for the five arm joints
        :param plan: use MoveIt to plan to the desired joint angles
        :param wait: wait until execution is finished

        :returns: True if successful; False otherwise
        """
        try:
            self.debug_print("set_joint_angles")
            joint_1, joint_2, joint_3, joint_4, joint_5 = joints
            resp = self.arm_joint_cmd_client(joint_1, joint_2, joint_3, joint_4, joint_5, plan, wait, check_obstruction)
            self.debug_print("done")
            return resp.result
        except rospy.ServiceException as e:
            traceback.print_exc(e)
            return False

    def set_end_effector_pose(self, xyz, pitch, roll=None, plan=False, wait=True, check_obstruction=False):
        """ Set desired pose (position+orientation) of the arm end-effector

        :param xyz: xyz position
        :param pitch: pitch angle
        :param roll: roll angle
        :param plan: use MoveIt to plan to the desired pose
        :param: wait until execution is finished

        :returns: True if successful; False otherwise
        """
        try:
            self.debug_print("set_end_effector_pose")
            x, y, z = xyz
            resp = self.arm_ee_cmd_client(x, y, z, pitch, roll, plan, wait, check_obstruction)
            self.debug_print("done")
            return resp.result
        except rospy.ServiceException as e:
            traceback.print_exc(e)
            return False

    def set_gripper_state(self, command, wait=True):
        """ Opens or closes the gripper

        :param command: passed as the command for the gripper
        :param wait: wait until execution is finished

        :returns: True if successful; False otherwise
        """
        try:
            self.debug_print("set_gripper_state", command)
            resp = self.gripper_cmd_client(command, wait)
            self.debug_print("done")
            return resp.result
        except rospy.ServiceException as e:
            traceback.print_exc(e)
            return False

    def open_gripper(self, wait=True):
        return self.set_gripper_state('open', wait=wait)

    def close_gripper(self, wait=True):
        return self.set_gripper_state('close', wait=wait)

    def force_close_if_gripper_open(self, wait=True):
        return self.set_gripper_state('force_close_if_open', wait=wait)

    def get_gripper_state(self):
        """ Fetches gripper state

        :returns: gripper state value (-1 if unknown)
            0: gripper fully open
            1: gripper closing
            2: object in the gripper
            3: gripper fully closed
        """
        return self._gripper_state.data

    def set_pan_tilt(self, pan, tilt, wait=True):
        """ Set pan-tilt angle of the camera

        :param pan: pan angle
        :param tilt: tilt angle
        :param wait: wait until execution is finished

        :returns: True if successful; False otherwise
        """
        try:
            self.debug_print("set_pan_tilt")
            resp = self.pan_tilt_client(pan, tilt, wait)
            self.debug_print("done")
            return resp.success
        except rospy.ServiceException as e:
            traceback.print_exc(e)
            return False

    def get_pan_tilt(self):
        joint_msg = self._joints
        pan_id = joint_msg.name.index('head_pan_joint')
        tilt_id = joint_msg.name.index('head_tilt_joint')
        return np.array([joint_msg.position[pan_id], joint_msg.position[tilt_id]])

    def get_joint_angles(self):
        joint_msg = self._joints
        angles = []
        for joint_name in JOINT_NAMES:
            joint_id = joint_msg.name.index(joint_name)
            angles += [joint_msg.position[joint_id]]
        return np.array(angles)

    def get_gripper_angle(self):
        joint_msg = self._joints
        joint_id = joint_msg.name.index(GRIPPER_JOINT_NAME)
        return joint_msg.position[joint_id]


    def get_fk_position(self):
        joint_positions = self.get_joint_angles().tolist()
        try:
            resp = self.fk_client(joint_positions)
        except rospy.ServiceException as e:
            traceback.print_exc(e)
            rospy.logerr("FK Service call failed")
            return None
        return resp.translation_rotation

    def get_ik_position(self, position, pitch, roll):
        try:
            resp = self.ik_client(position, pitch, roll)
        except rospy.ServiceException as e:
            traceback.print_exc(e)
            rospy.logerr("IK Service call failed")
            return None
        return resp.joint_positions

    def get_image(self):
        try:
            self.debug_print("get_image")
            resp = self.color_client()
            self.debug_print("done")
            return np.frombuffer(resp.image.data, dtype=np.uint8).reshape(480,640,-1)
        except rospy.ServiceException as e:
            traceback.print_exc(e)
            return False

    def get_depth(self):
        try:
            self.debug_print("get_image")
            resp = self.depth_client()
            self.debug_print("done")
            return np.frombuffer(resp.image.data, dtype=np.uint16).reshape(480,640,-1).astype(np.float32) / 1000
        except rospy.ServiceException as e:
            traceback.print_exc(e)
            return False

    def get_odom(self):
        pose = self._odom_state.pose.pose
        pos, ori = pose.position, pose.orientation
        pos = [pos.x, pos.y, pos.z]
        ori = [ori.x, ori.y, ori.z, ori.w]
        return pos, ori

    def get_pointcloud(self):
        try:
            resp = self.pointcloud_client(in_cam=False, filter_pts=False)
            print(len(resp.pts))
            return np.array(resp.pts).reshape((-1, 3))
        except rospy.ServiceException as e:
            traceback.print_exc(e)
            return False

    def is_grasp_obstructed(self, x, y):
        try:
            resp = self.grasp_obstructed_client(x=x, y=y)
            return resp.is_obstructed
        except rospy.ServiceException as e:
            traceback.print_exc(e)
            return True

    def get_bumper_state(self):
        return self._bumper_state

    def get_drop_state(self):
        return self._drop_state

    def get_power_state(self):
        return self._power_state

    def play_sound(self, sequence_id=4):
        msg = Sound()
        msg.value = sequence_id
        self.sound_publisher.publish(msg)

    def _odom_callback(self, msg):
        self._odom_state = msg

    def _gripper_callback(self, msg):
        self._gripper_state = msg

    def _joints_callback(self, msg):
        if len(msg.name) == 9:
            self._joints = msg

    def _bumper_callback(self, msg):
        self._bumper_state = msg.state

    def _drop_callback(self, msg):
        self._drop_state = msg.state

    def _power_callback(self, msg):
        self._power_state = msg.event
        if self._power_state == 5:
            print("Battery Critical")
            #if self._pause_filepath is None:
            rospy.signal_shutdown('Base battery power is critically low -- exiting...')
            #else:
            #    with open(self._pause_filepath, 'a'):
            #        os.utime(self._pause_filepath, None)


if __name__ == '__main__':
    interface = LocobotClient()
