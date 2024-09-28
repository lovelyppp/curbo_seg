#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
""" This example shows how to use cuRobo's kinematics to generate a mask. """


# Standard Library
import time

# Third Party
import imageio
import numpy as np
import torch
import torch.autograd.profiler as profiler
from nvblox_torch.datasets.mesh_dataset import MeshDataset
from torch.profiler import ProfilerActivity, profile, record_function

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import PointCloud, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.model.robot_segmenter import RobotSegmenter
import sys
from os.path import dirname

sys.path.append(dirname(dirname(__file__)))
from tools.realsenseD435i import RealsenseD435is
from tools.UR_Robot import UR_Robot

torch.manual_seed(30)

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def create_render_dataset(robot_file, robot):
    # load robot:
    robot_dict = load_yaml(join_path(get_robot_configs_path(), robot_file))
    robot_dict["robot_cfg"]["kinematics"]["load_link_names_with_mesh"] = True
    robot_dict["robot_cfg"]["kinematics"]["load_meshes"] = True

    robot_cfg = RobotConfig.from_dict(robot_dict["robot_cfg"])
    # q = [-2.97387964, -2.20343429, -1.20995075, -1.28440029, 1.65582216, 0.67572343]
    q = robot.get_current_joint()
    print(q)
    q = TensorDeviceType().to_device(q)
    kin_model = CudaRobotModel(robot_cfg.kinematics)
    q_js = JointState(position=q, joint_names=kin_model.joint_names)

    return q_js


def mask_image(robot: UR_Robot, robot_file="ur5e.yml"):
    save_debug_data = True
    write_pointcloud = True
    # create robot segmenter:
    tensor_args = TensorDeviceType()

    curobo_segmenter = RobotSegmenter.from_robot_file(
        robot_file,
        collision_sphere_buffer=0.1,
        distance_threshold=0.1,
        use_cuda_graph=True,
    )
    robot.camera.camera_config()
    robot.camera.wait_frames()
    color, depth = robot.camera.get_data()
    intrinsics = robot.camera.get_depth_intrinsics_matrix()
    pose = robot.camera.get_default_pose()

    q_js = create_render_dataset(robot_file, robot)

    if save_debug_data:
        visualize_scale = 10.0
        cam_obs = CameraObservation(
            depth_image=tensor_args.to_device(depth).unsqueeze(0)
            / robot.camera.depth_scale,
            intrinsics=tensor_args.to_device(intrinsics),
            pose=Pose.from_matrix(tensor_args.to_device(pose)),
        )
        # save depth image
        imageio.imwrite(
            "camera_depth.png",
            (cam_obs.depth_image * visualize_scale)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint16),
        )

        # save robot spheres in current joint configuration
        robot_kinematics = curobo_segmenter._robot_world.kinematics
        if write_pointcloud:
            sph = robot_kinematics.get_robot_as_spheres(q_js.position)
            WorldConfig(sphere=sph[0]).save_world_as_mesh("robot_spheres.stl")

            # save world pointcloud in robot origin

            pc = cam_obs.get_pointcloud()
            pc_obs = PointCloud("world", pose=cam_obs.pose.to_list(), points=pc)
            pc_obs.save_as_mesh("camera_pointcloud.stl", transform_with_pose=True)

        # run segmentation:
        depth_mask, filtered_image = curobo_segmenter.get_robot_mask_from_active_js(
            cam_obs,
            q_js,
        )
        # save robot points as mesh

        robot_mask = cam_obs.clone()
        robot_mask.depth_image[~depth_mask] = 0.0

        if write_pointcloud:
            robot_mesh = PointCloud(
                "world",
                pose=robot_mask.pose.to_list(),
                points=robot_mask.get_pointcloud(),
            )
            robot_mesh.save_as_mesh("robot_segmented.stl", transform_with_pose=True)
        # save depth image
        imageio.imwrite(
            "robot_depth.png",
            (robot_mask.depth_image * visualize_scale)
            .detach()
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.uint16),
        )

        # save world points as mesh

        world_mask = cam_obs.clone()
        world_mask.depth_image[depth_mask] = 0.0
        if write_pointcloud:
            world_mesh = PointCloud(
                "world",
                pose=world_mask.pose.to_list(),
                points=world_mask.get_pointcloud(),
            )
            world_mesh.save_as_mesh("world_segmented.stl", transform_with_pose=True)

        imageio.imwrite(
            "world_depth.png",
            (world_mask.depth_image * visualize_scale)
            .detach()
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.uint16),
        )
        robot.camera.stop()


if __name__ == "__main__":
    robot = UR_Robot(is_use_robotiq85=False, camera_id_list=[0])
    mask_image(
        robot,
        robot_file="../configs/ur3_robotiq_2f_85.yml",
    )
    # batch_mask_image(
    #     robot,
    #     robot_file="/home/ppppppp/ReKep/sdf_reconstruction/configs/ur3_robotiq_2f_85.yml",
    # )
