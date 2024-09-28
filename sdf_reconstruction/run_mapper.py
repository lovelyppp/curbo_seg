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
import torch
import demo_utils as demo_utils
from timer import Timer
from nvblox_torch.mapper import (
    Mapper,
)  # create_nvblox_mapper_instance, get_nvblox_mapper_class
from voxel_visualizer import SimpleVoxelVisualizer

import sys
from os.path import dirname

sys.path.append(dirname(dirname(__file__)))
from tools.realsenseD435i import RealsenseD435is


"""
This script implements similar functionality to fuse_3dmatch.cpp of nvblox, except through Python.
"""
from demo_utils import (
    global_pose,
    query_sdf_on_a_plane_grid,
    show_inputs,
    show_renders,
    show_voxels,
)


def get_sdf(camera: RealsenseD435is, voxel_size=0.03, visualize=0, max_frames=10):
    # Initialize NvBlox with 2 mappers: mapper 0 uses tsdf layer, mapper 1 uses occupancy
    mapper = Mapper([voxel_size, voxel_size], ["tsdf", "occupancy"])
    mapper_id = 0

    timer = Timer()

    if visualize:
        visualizer = SimpleVoxelVisualizer(voxel_size)

    timer.start("data")
    camera.camera_config()
    num_frame = 0
    while True:
        camera.wait_frames()
        for i in range(len(camera.camera_id_list)):
            rgb_np, depth_np = camera.get_data(camera_id=i)
            pose_np = camera.get_default_pose()
            intrinsics_np = camera.get_color_intrinsics_matrix()

            rgb = torch.from_numpy(rgb_np).permute((2, 0, 1))
            rgba = torch.cat([rgb, torch.ones_like(rgb[0:1, :, :]) * 255], dim=0)
            depth = torch.from_numpy(depth_np).float()
            pose = torch.from_numpy(pose_np).float()
            intrinsics = torch.from_numpy(intrinsics_np).float()

            rgba, depth, pose, intrinsics = demo_utils.prep_inputs(
                rgba, depth, pose, intrinsics
            )

            timer.end("data")

            timer.start("update")

            mapper.add_depth_frame(depth, pose, intrinsics, mapper_id)
            mapper.update_esdf(-1)

            timer.end("update")

            if visualize:
                show_inputs(rgba, depth)
                # Rendering is only supported by the tsdf mapper
                show_renders(mapper, pose, "current_pose", timer, mapper_id=0)
                show_renders(mapper, global_pose(), "global_pose", timer, mapper_id=0)
                show_voxels(visualizer, mapper)
                query_sdf_on_a_plane_grid("combined_sdf", mapper, timer, mapper_id=-1)
                query_sdf_on_a_plane_grid("static_sdf", mapper, timer, mapper_id=0)
                query_sdf_on_a_plane_grid("dynamic_sdf", mapper, timer, mapper_id=1)

            timer.print()
            timer.start("data")
        num_frame += 1
        if num_frame > max_frames:
            break

    timer.end("data")
    camera.stop()

    if visualize:
        visualizer.end()

    timer.print()
    return get_esdf(voxel_size, mapper)


def get_esdf(voxel_size, mapper: Mapper):
    tensor_args = {"device": "cuda", "dtype": torch.float32}

    low = [-5, -5, -1]
    high = [5, 5, 2]
    range = [h - l for l, h in zip(low, high)]

    x = torch.linspace(low[0], high[0], int(range[0] // voxel_size), **tensor_args)
    y = torch.linspace(low[1], high[1], int(range[1] // voxel_size), **tensor_args)
    z = torch.linspace(low[2], high[2], int(range[2] // voxel_size), **tensor_args)
    xyz = (
        torch.stack(torch.meshgrid(x, y, z, indexing="ij"))
        .permute((1, 2, 3, 0))
        .reshape(-1, 3)
    )

    r = torch.zeros_like(xyz[:, 0:1])  # * voxel_size
    xyzr = torch.cat([xyz, r], dim=1)
    batch_size = xyzr.shape[0]

    out_points = torch.zeros((batch_size, 4), **tensor_args) + 0.0

    return mapper.query_sdf(xyzr, out_points, True, mapper_id=-1)


if __name__ == "__main__":
    cap = RealsenseD435is(camera_id_list=[0])
    get_sdf(cap)
