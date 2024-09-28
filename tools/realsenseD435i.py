import pyrealsense2 as rs
import cv2
import numpy as np
from transforms3d import affines, euler
import yaml
from scipy.spatial.transform import Rotation as R


class RealsenseD435is:

    def __init__(
        self,
        camera_id_list=[0],
        camera_width=1280,
        camera_height=720,
        camera_fps=30,
    ):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.camera_id_list = camera_id_list
        self.ALIGN_WAY = 1  # 0:彩色图像对齐到深度图;1:深度图对齐到彩色图像
        # filter tag
        self.tag_filter = 1

        with open(
            "/home/ppppppp/ReKep/sdf_reconstruction/configs/camera_configs.yaml", "r"
        ) as file:
            data = yaml.safe_load(file)
            self.trans = data["transformation"]
            self.color_intrinsics = data["color_intrinsic"]
            self.left_depth_intrinsics = data["depth_intrinsic"]["left"]
            self.right_depth_intrinsics = data["depth_intrinsic"]["right"]

    def camera_config(self):
        self.connect_device = []
        # get all realsense serial number
        for d in rs.context().devices:
            print(
                "Found device: ",
                d.get_info(rs.camera_info.name),
                " ",
                d.get_info(rs.camera_info.serial_number),
            )
            if d.get_info(rs.camera_info.name).lower() != "platform camera":
                self.connect_device.append(d.get_info(rs.camera_info.serial_number))
        # config realsense
        self.pipeline_list = [rs.pipeline() for i in range(len(self.camera_id_list))]
        self.config_list = [rs.config() for i in range(len(self.camera_id_list))]
        self.profile_list = [None for i in range(len(self.camera_id_list))]
        if len(self.camera_id_list) == 1:  # one realsense
            self.config_list[0].enable_device(self.connect_device[0])
            self.config_list[0].enable_stream(
                rs.stream.depth,
                self.camera_width,
                self.camera_height,
                rs.format.z16,
                self.camera_fps,
            )
            self.config_list[0].enable_stream(
                rs.stream.color,
                self.camera_width,
                self.camera_height,
                rs.format.bgr8,
                self.camera_fps,
            )
            self.profile_list[0] = self.pipeline_list[0].start(self.config_list[0])
            self.intensity_setpoint(0, 600)

        elif len(self.camera_id_list) >= 2:  # two realsense
            if len(self.connect_device) < 2:
                print(
                    "Registrition needs two camera connected.But got one.Please run realsense-viewer to check your camera status."
                )
                exit()
            # enable config
            for n, config in enumerate(self.config_list):
                config.enable_device(self.connect_device[n])
                config.enable_stream(
                    rs.stream.depth,
                    self.camera_width,
                    self.camera_height,
                    rs.format.z16,
                    self.camera_fps,
                )
                config.enable_stream(
                    rs.stream.color,
                    self.camera_width,
                    self.camera_height,
                    rs.format.bgr8,
                    self.camera_fps,
                )
                # self.config_list[n] = config
            # start pipeline
            for n, pipeline in enumerate(self.pipeline_list):
                self.profile_list[n] = pipeline.start(self.config_list[n])
                self.intensity_setpoint(n, 600)

    def intensity_setpoint(self, frame_id, meanintensity=600):
        advnc_mode = rs.rs400_advanced_mode(self.profile_list[frame_id].get_device())
        ae_ctrl = advnc_mode.get_ae_control()
        ae_ctrl.meanIntensitySetPoint = meanintensity
        advnc_mode.set_ae_control(ae_ctrl)

    def wait_frames(self, frame_id=None):
        """
        camera_id:
            @ = camera number , get all frame
            @ = id , get specific id camera frame
        """
        self.frame_list = [None for i in range(len(self.camera_id_list))]
        if frame_id != None:  # give a frame id
            self.frame_list[frame_id] = self.pipeline_list[frame_id].wait_for_frames()
        else:  # get all frame
            if len(self.camera_id_list) == 1:
                self.frame_list[0] = self.pipeline_list[0].wait_for_frames()
            else:
                for n, camera_id in enumerate(self.camera_id_list):
                    self.frame_list[n] = self.pipeline_list[n].wait_for_frames()

    def multi_filter(self, frame, clipping_distance):
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)

        decimation = rs.decimation_filter()
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        # 0: fill from left 1:farest from around 2:nearest from around
        hole_filling = rs.hole_filling_filter(2)
        threshold_filter = rs.threshold_filter()  # Threshold
        threshold_filter.set_option(rs.option.min_distance, 0.05)
        threshold_filter.set_option(rs.option.max_distance, clipping_distance)

        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.6)
        spatial.set_option(rs.option.filter_smooth_delta, 8)
        spatial.set_option(rs.option.holes_fill, 5)

        # frame = decimation.process(frame)
        frame = depth_to_disparity.process(frame)
        frame = spatial.process(frame)
        frame = temporal.process(frame)
        frame = disparity_to_depth.process(frame)
        # frame = hole_filling.process(frame)
        frame = threshold_filter.process(frame)
        return frame

    def get_data(self, camera_id=0, clipping_distance_m=2):
        if self.ALIGN_WAY:
            way = rs.stream.color
        else:
            way = rs.stream.depth
        align = rs.align(way)

        depth_sensor = self.profile_list[camera_id].get_device().first_depth_sensor()

        # 00: Custom 01: Default 02: Hand 03: High Accuracy 04: High Density
        # preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        # for i in range(int(preset_range.max)):
        #     visulpreset = depth_sensor.get_option_value_description(
        #         rs.option.visual_preset, i
        #     )
        #     print("%02d: %s" % (i, visulpreset))
        depth_sensor.set_option(rs.option.visual_preset, 4)

        self.depth_scale = depth_sensor.get_depth_scale()
        aligned_frames = align.process(self.frame_list[camera_id])
        depth_frame_aligned = aligned_frames.get_depth_frame()
        color_frame_aligned = aligned_frames.get_color_frame()
        # filter
        if self.tag_filter == 1:
            depth_frame_aligned = self.multi_filter(
                depth_frame_aligned, clipping_distance_m
            )
            for i in range(5):
                self.wait_frames(camera_id)
                aligned_frames = align.process(self.frame_list[camera_id])
                depth_frame_aligned = aligned_frames.get_depth_frame()
                depth_frame_aligned = self.multi_filter(
                    depth_frame_aligned, clipping_distance_m
                )

        color_image_aligned = np.asanyarray(color_frame_aligned.get_data())
        depth_image_aligned = np.asanyarray(depth_frame_aligned.get_data())

        depth_image_aligned = depth_image_aligned.astype(np.float32) * self.depth_scale

        return color_image_aligned, depth_image_aligned

    def rgb_image(self, camera_id=0):
        color_frame = self.frame_list[camera_id].get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def depth_image(self, camera_id=0, clipping_distance_m=2.5):
        depth_sensor = self.profile_list[camera_id].get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 4)

        depth_scale = depth_sensor.get_depth_scale()
        # clipping_distance = clipping_distance_m / depth_scale
        depth_frame = self.frame_list[camera_id].get_depth_frame()
        # filter
        if self.tag_filter == 1:
            depth_frame = self.multi_filter(depth_frame, clipping_distance_m)
            for i in range(5):
                self.wait_frames(camera_id)
                depth_frame = self.frame_list[camera_id].get_depth_frame()
                depth_frame = self.multi_filter(depth_frame, clipping_distance_m)
        depth_image = np.asanyarray(depth_frame.get_data())
        # depth_image[depth_image > clipping_distance] = 0.0
        depth_image = depth_image.astype(np.float32) * depth_scale

        return depth_image

    def get_color_intrinsics_matrix(self):
        intrinsics = self.color_intrinsics
        intrinsic_matrix = np.array(
            [
                [intrinsics["fx"], 0, intrinsics["ppx"]],
                [0, intrinsics["fy"], intrinsics["ppy"]],
                [0, 0, 1],
            ]
        )
        return intrinsic_matrix

    def get_depth_intrinsics_matrix(self):
        intrinsic_matrix = np.array(
            [
                [
                    (
                        self.right_depth_intrinsics["fx"]
                        + self.left_depth_intrinsics["fx"]
                    )
                    / 2,
                    0,
                    (
                        self.right_depth_intrinsics["ppx"]
                        + self.left_depth_intrinsics["ppx"]
                    )
                    / 2,
                ],
                [
                    0,
                    (
                        self.right_depth_intrinsics["fy"]
                        + self.left_depth_intrinsics["fy"]
                    )
                    / 2,
                    (
                        self.right_depth_intrinsics["ppy"]
                        + self.left_depth_intrinsics["ppy"]
                    )
                    / 2,
                ],
                [0, 0, 1],
            ]
        )
        return intrinsic_matrix

    def get_default_pose(self):
        quaternion = (
            self.trans["qx"],
            self.trans["qy"],
            self.trans["qz"],
            self.trans["qw"],
        )
        p = [self.trans["x"], self.trans["y"], self.trans["z"]]
        Rot = R.from_quat(quaternion).as_matrix()
        T = affines.compose(p, Rot, [1, 1, 1])
        return np.array(T)

    def stop(self):
        for pipeline in self.pipeline_list:
            pipeline.stop()
        print("camera exit sucessfully.")


if __name__ == "__main__":
    cap = RealsenseD435is(camera_id_list=[0], camera_width=1280, camera_height=720)  #
    cap.camera_config()
    i = 0
    while True:
        cap.wait_frames()
        # img0 = cap.rgb_image(0)
        # cv2.imshow("img0", img0)
        img1, depth0 = cap.get_data()
        depth1 = cap.depth_image()
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth1 * 1000, alpha=0.03), cv2.COLORMAP_JET
        )
        images = np.hstack((img1, depth_colormap))
        # Show images
        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RealSense", images)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyAllWindows()
            break
    cap.stop()
