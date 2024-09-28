# coding=utf8
import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import scipy as sc
from collections import namedtuple

# import utils
import socket
import select
import struct
import numpy as np
import math
from tools.robotiq_gripper import RobotiqGripper
from tools.realsenseD435i import RealsenseD435is


class UR_Robot:

    def __init__(
        self,
        tcp_host_ip="192.168.1.102",
        tcp_port=30003,
        workspace_limits=None,
        is_use_robotiq85=True,
        is_use_camera=True,
        camera_id_list=[0],
    ):
        # Init varibles
        if workspace_limits is None:
            workspace_limits = [[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]
        self.tcp_host_ip = tcp_host_ip
        self.tcp_port = tcp_port
        self.is_use_robotiq85 = is_use_robotiq85
        self.is_use_camera = is_use_camera
        self.workspace_limits = workspace_limits
        self.camera_id_list = camera_id_list
        # self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # UR5 robot configuration
        # Default joint/tool speed configuration
        self.joint_acc = 1.4  # Safe: 1.4   8
        self.joint_vel = 1.05  # Safe: 1.05  3

        # Joint tolerance for blocking calls
        self.joint_tolerance = 0.01

        # Default tool speed configuration
        self.tool_acc = 0.5  # Safe: 0.5
        self.tool_vel = 0.2  # Safe: 0.2

        # Tool pose tolerance for blocking calls
        self.tool_pose_tolerance = [0.003, 0.003, 0.003, 0.01, 0.01, 0.01]

        # robotiq85 gripper configuration
        if self.is_use_robotiq85:
            # reference https://gitlab.com/sdurobotics/ur_rtde
            # Gripper activate
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.tcp_host_ip, 63352)  # don't change the 63352 port
            self.gripper._reset()
            print("Activating gripper...")
            self.gripper.activate()
            time.sleep(1.5)

        # realsense configuration
        if self.is_use_camera:
            # Fetch RGB-D data from RealSense camera
            self.camera = RealsenseD435is(camera_id_list=self.camera_id_list)
            # # Load camera pose (from running calibrate.py), intrinsics and depth scale
            # self.cam_pose = np.loadtxt('real/camera_pose.txt', delimiter=' ')
            # self.cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')

        # Default robot home joint configuration (the robot is up to air)
        self.home_joint_config = [
            -(0 / 360.0) * 2 * np.pi,
            -(90 / 360.0) * 2 * np.pi,
            (0 / 360.0) * 2 * np.pi,
            -(90 / 360.0) * 2 * np.pi,
            -(0 / 360.0) * 2 * np.pi,
            0.0,
        ]

        # test
        # self.get_camera_data()
        # self.testRobot()

    # Test for robot control
    def testRobot(self):
        try:
            print("Test for robot...")
            # self.move_j([-(0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
            #                  (0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
            #                  -(0 / 360.0) * 2 * np.pi, 0.0])
            # self.move_j([(57.04 / 360.0) * 2 * np.pi, (-65.26/ 360.0) * 2 * np.pi,
            #                  (73.52/ 360.0) * 2 * np.pi, (-100.89/ 360.0) * 2 * np.pi,
            #                  (-86.93/ 360.0) * 2 * np.pi, (-0.29/360)*2*np.pi])
            print("11")
            # self.open_gripper()
            # self.move_j([(57.03 / 360.0) * 2 * np.pi, (-56.67 / 360.0) * 2 * np.pi,
            #                   (88.72 / 360.0) * 2 * np.pi, (-124.68 / 360.0) * 2 * np.pi,
            #                   (-86.96/ 360.0) * 2 * np.pi, (-0.3/ 360) * 2 * np.pi])
            # self.close_gripper()
            # self.move_j([(57.04 / 360.0) * 2 * np.pi, (-65.26 / 360.0) * 2 * np.pi,
            #                   (73.52 / 360.0) * 2 * np.pi, (-100.89 / 360.0) * 2 * np.pi,
            #                   (-86.93 / 360.0) * 2 * np.pi, (-0.29 / 360) * 2 * np.pi])

            # self.move_j([-(0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
            #                  (0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
            #                  -(0 / 360.0) * 2 * np.pi, 0.0])
            # self.move_j_p([-0.266, -0.618, 0.541, 0.872, 2.976, 0])
            # self.move_p([-0.266, -0.618, 0.441, 0.872, 2.976, 0])
            # self.move_l([-0.266, -0.618, 0.441, 0.872, 2.976, 0.089])
            print("22")
            # move_c bug
            # self.move_c([-0.366, -0.518, 0.441, 0.872, 2.976, 0],[-0.266, -0.418, 0.441, 0.872, 2.976, 0])
        except:
            print("Test fail! Please check the ip address or integrity of the file")

    # joint control
    """
    input:joint_configuration = joint angle
    """

    def move_j(self, joint_configuration, k_acc=1, k_vel=1, t=0, r=0):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        # command: movej([joint_configuration],a,v,t,r)\n
        tcp_command = "movej([%f" % joint_configuration[0]  # "movej([]),a=,v=,\n"
        for joint_idx in range(1, 6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f,t=%f,r=%f)\n" % (
            k_acc * self.joint_acc,
            k_vel * self.joint_vel,
            t,
            r,
        )
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(1500)
        actual_joint_positions = self.parse_tcp_state_data(state_data, "joint_data")
        while not all(
            [
                np.abs(actual_joint_positions[j] - joint_configuration[j])
                < self.joint_tolerance
                for j in range(6)
            ]
        ):
            state_data = self.tcp_socket.recv(1500)
            actual_joint_positions = self.parse_tcp_state_data(state_data, "joint_data")
            time.sleep(0.01)
        self.tcp_socket.close()
        return True

    # joint control
    """
    move_j_p(self, tool_configuration,k_acc=1,k_vel=1,t=0,r=0)
    input:tool_configuration=[x y z r p y]
    其中x y z为三个轴的目标位置坐标，单位为米
    r p y ，单位为弧度
    """

    def move_j_p(self, tool_configuration, k_acc=1, k_vel=1, t=0, r=0):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        print(f"movej_p([{tool_configuration}])")

        # tcp_command = "def process():\n"
        # tcp_command += " array = rpy2rotvec([%f,%f,%f])\n" % (
        #     tool_configuration[3],
        #     tool_configuration[4],
        #     tool_configuration[5],
        # )
        # tcp_command += "movej(get_inverse_kin(p[%f,%f,%f,array[0],array[1],array[2]]),a=%f,v=%f,t=%f,r=%f)\n" % (
        #     tool_configuration[0],
        #     tool_configuration[1],
        #     tool_configuration[2],
        #     k_acc * self.joint_acc,
        #     k_vel * self.joint_vel,
        #     t,
        #     r,
        # )  # "movej([]),a=,v=,\n"
        # tcp_command += "end\n"

        tcp_command = (
            "movej(get_inverse_kin(p[%f" % tool_configuration[0]
        )  # "movej([]),a=,v=,\n"
        for joint_idx in range(1, 6):
            tcp_command = tcp_command + (",%f" % tool_configuration[joint_idx])
        tcp_command = tcp_command + "]),a=%f,v=%f)\n" % (
            k_acc * self.joint_acc,
            k_vel * self.joint_vel,
        )

        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        start = time.time()
        delaytime = 100
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, "cartesian_info")
        while not all(
            [
                np.abs(actual_tool_positions[j] - tool_configuration[j])
                < self.tool_pose_tolerance[j]
                for j in range(3)
            ]
        ) and (time.time() - start < delaytime):
            state_data = self.tcp_socket.recv(1500)
            # print(f"tool_position_error{actual_tool_positions - tool_configuration}")
            actual_tool_positions = self.parse_tcp_state_data(
                state_data, "cartesian_info"
            )
            time.sleep(0.01)
        if (time.time() - start) >= delaytime:
            print(f"{time.time()},start:{start}")
            return False
        self.tcp_socket.close()
        time.sleep(0.5)
        return True

    # Usually, We don't use move_p
    # move_p is mean that the robot keep the same speed moving
    def move_p(self, tool_configuration, k_acc=1, k_vel=1, r=0):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        print(f"movep([{tool_configuration}])")
        # command: movep([tool_configuration],a,v,t,r)\n
        tcp_command = "def process():\n"
        tcp_command += " array = rpy2rotvec([%f,%f,%f])\n" % (
            tool_configuration[3],
            tool_configuration[4],
            tool_configuration[5],
        )
        tcp_command += (
            "movep(p[%f,%f,%f,array[0],array[1],array[2]],a=%f,v=%f,r=%f)\n"
            % (
                tool_configuration[0],
                tool_configuration[1],
                tool_configuration[2],
                k_acc * self.joint_acc,
                k_vel * self.joint_vel,
                r,
            )
        )  # "movep([]),a=,v=,\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, "cartesian_info")
        while not all(
            [
                np.abs(actual_tool_positions[j] - tool_configuration[j])
                < self.tool_pose_tolerance[j]
                for j in range(3)
            ]
        ):
            state_data = self.tcp_socket.recv(1500)
            actual_tool_positions = self.parse_tcp_state_data(
                state_data, "cartesian_info"
            )
            time.sleep(0.01)
        time.sleep(1.5)
        self.tcp_socket.close()

    # move_l is mean that the robot keep running in a straight line
    def move_l(self, tool_configuration, k_acc=1, k_vel=1, t=0, r=0):
        print(f"movel([{tool_configuration}])")
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        # command: movel([tool_configuration],a,v,t,r)\n
        tcp_command = "def process():\n"
        tcp_command += " array = rpy2rotvec([%f,%f,%f])\n" % (
            tool_configuration[3],
            tool_configuration[4],
            tool_configuration[5],
        )
        tcp_command += (
            "movel(p[%f,%f,%f,array[0],array[1],array[2]],a=%f,v=%f,t=%f,r=%f)\n"
            % (
                tool_configuration[0],
                tool_configuration[1],
                tool_configuration[2],
                k_acc * self.joint_acc,
                k_vel * self.joint_vel,
                t,
                r,
            )
        )  # "movel([]),a=,v=,\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        start = time.time()
        delaytime = 10
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, "cartesian_info")
        while (
            not all(
                [
                    np.abs(actual_tool_positions[j] - tool_configuration[j])
                    < self.tool_pose_tolerance[j]
                    for j in range(3)
                ]
            )
            and time.time() - start < delaytime
        ):
            state_data = self.tcp_socket.recv(1500)
            actual_tool_positions = self.parse_tcp_state_data(
                state_data, "cartesian_info"
            )
            time.sleep(0.01)
        if (time.time() - start) >= delaytime:
            return False
        time.sleep(0.5)
        self.tcp_socket.close()
        return True

    # Usually, We don't use move_c
    # move_c is mean that the robot move circle
    # mode 0: Unconstrained mode. Interpolate orientation from current pose to target pose (pose_to)
    #      1: Fixed mode. Keep orientation constant relative to the tangent of the circular arc (starting from current pose)
    def move_c(self, pose_via, tool_configuration, k_acc=1, k_vel=1, r=0, mode=0):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        print(f"movec([{pose_via},{tool_configuration}])")
        # command: movec([pose_via,tool_configuration],a,v,t,r)\n
        tcp_command = "def process():\n"
        tcp_command += " via_pose = rpy2rotvec([%f,%f,%f])\n" % (
            pose_via[3],
            pose_via[4],
            pose_via[5],
        )
        tcp_command += " tool_pose = rpy2rotvec([%f,%f,%f])\n" % (
            tool_configuration[3],
            tool_configuration[4],
            tool_configuration[5],
        )
        tcp_command = f" movec([{pose_via[0]},{pose_via[1]},{pose_via[2]},via_pose[0],via_pose[1],via_pose[2]], \
                [{tool_configuration[0]},{tool_configuration[1]},{tool_configuration[2]},tool_pose[0],tool_pose[1],tool_pose[2]], \
                a={k_acc * self.tool_acc},v={k_vel * self.tool_vel},r={r})\n"
        tcp_command += "end\n"

        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, "cartesian_info")
        while not all(
            [
                np.abs(actual_tool_positions[j] - tool_configuration[j])
                < self.tool_pose_tolerance[j]
                for j in range(3)
            ]
        ):
            state_data = self.tcp_socket.recv(1500)
            actual_tool_positions = self.parse_tcp_state_data(
                state_data, "cartesian_info"
            )
            time.sleep(0.01)
        self.tcp_socket.close()
        time.sleep(1.5)

    def go_home(self):
        self.move_j(self.home_joint_config)

    def restartReal(self):
        self.go_home()
        # robotiq85 gripper configuration
        if self.is_use_robotiq85:
            # reference https://gitlab.com/sdurobotics/ur_rtde
            # Gripper activate
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.tcp_host_ip, 63352)  # don't change the 63352 port
            self.gripper._reset()
            print("Activating gripper...")
            self.gripper.activate()
            time.sleep(1.5)

        # realsense configuration
        if self.is_use_camera:
            # Fetch RGB-D data from RealSense camera
            self.camera = RealsenseD435is(camera_id_list=self.camera_id_list)
            # # Load camera pose (from running calibrate.py), intrinsics and depth scale
            # self.cam_pose = np.loadtxt("real/camera_pose.txt", delimiter=" ")
            # self.cam_depth_scale = np.loadtxt("real/camera_depth_scale.txt", delimiter=" ")

    # get robot current state and information
    def get_state(self):
        self.tcp_cket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        state_data = self.tcp_socket.recv(1500)
        self.tcp_socket.close()
        return state_data

    # get robot current joint angles and cartesian pose
    def parse_tcp_state_data(self, state_data, subpackage):
        dic = {
            "MessageSize": "i",
            "Time": "d",
            "q target": "6d",
            "qd target": "6d",
            "qdd target": "6d",
            "I target": "6d",
            "M target": "6d",
            "q actual": "6d",
            "qd actual": "6d",
            "I actual": "6d",
            "I control": "6d",
            "Tool vector actual": "6d",
            "TCP speed actual": "6d",
            "TCP force": "6d",
            "Tool vector target": "6d",
            "TCP speed target": "6d",
            "Digital input bits": "d",
            "Motor temperatures": "6d",
            "Controller Timer": "d",
            "Test value": "d",
            "Robot Mode": "d",
            "Joint Modes": "6d",
            "Safety Mode": "d",
            "empty1": "6d",
            "Tool Accelerometer values": "3d",
            "empty2": "6d",
            "Speed scaling": "d",
            "Linear momentum norm": "d",
            "SoftwareOnly": "d",
            "softwareOnly2": "d",
            "V main": "d",
            "V robot": "d",
            "I robot": "d",
            "V actual": "6d",
            "Digital outputs": "d",
            "Program state": "d",
            "Elbow position": "3d",
            "Elbow velocity": "3d",
            "Safety Status": "d",
        }
        ii = range(len(dic))
        for key, i in zip(dic, ii):
            fmtsize = struct.calcsize(dic[key])
            data1, state_data = state_data[0:fmtsize], state_data[fmtsize:]
            fmt = "!" + dic[key]
            dic[key] = dic[key], struct.unpack(fmt, data1)

        if subpackage == "joint_data":  # get joint data
            q_actual_tuple = dic["q actual"]
            joint_data = np.array(q_actual_tuple[1])
            return joint_data
        elif subpackage == "cartesian_info":
            Tool_vector_actual = dic["Tool vector actual"]  # get x y z rx ry rz
            cartesian_info = np.array(Tool_vector_actual[1])
            return cartesian_info

    def get_current_pos(self):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, "cartesian_info")
        time.sleep(0.01)
        self.tcp_socket.close()
        return actual_tool_positions

    def get_current_joint(self):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        state_data = self.tcp_socket.recv(1500)
        actual_joint_positions = self.parse_tcp_state_data(state_data, "joint_data")
        time.sleep(0.01)
        self.tcp_socket.close()
        return actual_joint_positions

    def move_openvla(self, delta_action, speed=255, force=125):
        curr_tcp = self.get_current_pos()
        new_tcp = curr_tcp + np.array(delta_action[0:6])
        print("current tcp:", curr_tcp, "\n", "new tcp:", new_tcp)
        # self.move_p(new_tcp)
        self.move_j_p(new_tcp)

        delta_tool = int((1.00 - delta_action[6]) * 255.00)
        if abs(delta_tool) > 20:
            curr_tool = self.gripper.get_current_position()
            new_tool = curr_tool + delta_tool
            print("gripper is moving:", new_tool)
            self.gripper.move_and_wait_for_pos(new_tool, speed, force)

    ## robotiq85 gripper
    # get gripper position [0-255]  open:0 ,close:255
    def get_current_tool_pos(self):
        return self.gripper.get_current_position()

    def log_gripper_info(self, gripper):
        print(
            f"Pos: {str(gripper.get_current_position())}  "
            f"Open: {gripper.is_open()} "
            f"Closed: {gripper.is_closed()} "
        )

    def close_gripper(self, speed=255, force=255):
        # position: int[0-255], speed: int[0-255], force: int[0-255]
        self.gripper.move_and_wait_for_pos(255, speed, force)
        self.log_gripper_info(self.gripper)
        print("gripper had closed!")
        time.sleep(1.2)

    def open_gripper(self, speed=255, force=255):
        # position: int[0-255], speed: int[0-255], force: int[0-255]
        self.gripper.move_and_wait_for_pos(0, speed, force)
        self.log_gripper_info(self.gripper)
        print("gripper had opened!")
        time.sleep(1.2)

    ## get camera data
    def get_camera_data(self):
        return self.camera.get_color_pil()

    # Note: must be preceded by close_gripper()
    def check_grasp(self):
        # if the robot grasp object ,then the gripper is not open
        return self.get_current_tool_pos() > 220


if __name__ == "__main__":
    ur_robot = UR_Robot()
