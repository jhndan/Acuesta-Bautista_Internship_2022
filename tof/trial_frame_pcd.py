#
# BSD 3-Clause License
#
# Copyright (c) 2019, Analog Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import sys,os
from unittest import result
sys.path.append(os.path.join(os.getcwd(), '.'))
import aditofpython.aditofpython as tof
import numpy as np
import cv2 as cv
import open3d as o3d
from enum import Enum
import time
import keyboard
import matplotlib.pyplot as plt

WINDOW_NAME_DEPTH = "Display Depth"
WINDOW_NAME_COLOR = "Display Color"
configFile = "aditofpython/tof-viewer_config.json"


class ModesEnum(Enum):
    MODE_NEAR = 0
    MODE_MEDIUM = 1
    MODE_FAR = 2


def transform_image(np_image):
    return o3d.geometry.Image(np_image)


if __name__ == "__main__":
    system = tof.System()
    
    
    cameras = []
    status = system.getCameraListAtIp(cameras,"10.42.0.1")
    if not status:
        print("system.getCameraList() failed with status: ", status)
        
    status = cameras[0].setControl("initialization_config", configFile)
    if not status:
        print("cameras[0].setControl() failed with status: ", status)

    status = cameras[0].initialize()
    if not status:
        print("cameras[0].initialize() failed with status: ", status)


    modes = []
    status = cameras[0].getAvailableModes(modes)
    if not status:
        print ("system.getAvailableModes() fail with status",status)

    status = cameras[0].setMode(modes[ModesEnum.MODE_MEDIUM.value])
    if not status:
        print("cameras[0].setMode() failed with status: ", status)

    types = []
    status = cameras[0].getAvailableFrameTypes(types)
    if not status:
        print("system.getAvailableFrameTypes() failed with status: ", status)

    status = cameras[0].setFrameType(types[0])
    if not status:
        print("cameras[0].setFrameType() failed with status:", status)

    status = cameras[0].start()
    if not status:
        print("cameras[0].start() failed with status:", status)
    # status = cameras[0].setMode(modes[ModesEnum.MODE_NEAR.value])
    # if not status:
    #     print("cameras[0].setMode() failed with status: ", status)

    camDetails = tof.CameraDetails()
    # print (f"\nthe cameraDetails : {camDetails}\n")
    status = cameras[0].getDetails(camDetails)
    if not status:
        print("system.getDetails() failed with status: ", status)

    # Enable noise reduction for better results
    smallSignalThreshold = 100
    cameras[0].setControl("noise_reduction_threshold", str(smallSignalThreshold))

    # Get the first frame for details
    camera_range = 1000
    bitCount = 9
    frame = tof.Frame()
    status = cameras[0].requestFrame(frame)
    frameDataDetails = tof.FrameDataDetails()
    status = frame.getDataDetails("depth", frameDataDetails)
    width = frameDataDetails.width
    height = frameDataDetails.height
    

    # Get intrinsic parameters from camera
    intrinsicParameters = camDetails.intrinsics
    fx = intrinsicParameters.fx
    
    fy = intrinsicParameters.fy
    
    cx = intrinsicParameters.cx

    cy = intrinsicParameters.cy
    
    cameraIntrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    

    # Get camera details for frame correction
    # camera_range = camDetails.maxDepth
    print (f"\nthe camera range : {camera_range}\n")
    # bitCount = camDetails.bitCount
    print (f"\nthe bitCount : {bitCount}\n")
    max_value_of_IR_pixel = 2 ** bitCount - 1
    print (f"\nthe max value of ir pixel : {max_value_of_IR_pixel}\n")
    distance_scale_ir = 255.0 / max_value_of_IR_pixel
    print (f"\n{distance_scale_ir}\n")
    distance_scale = 255.0 / camera_range

    # Create visualizer for depth and ir
    # vis_depth = o3d.visualization.Visualizer()
    # vis_depth.create_window("Depth", 2 * width, 2 * height)

    # vis_ir = o3d.visualization.Visualizer()
    # vis_ir.create_window("IR", 2 * width, 2 * height)

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    # vis.create_window("PointCloud", 1200, 1200)
    # first_time_render_pc = 1
    # point_cloud = o3d.geometry.PointCloud()

   
        # Capture frame-by-frame
    status = cameras[0].requestFrame(frame)
    if not status:
        print("cameras[0].requestFrame() failed with status: ", status)

    depth_map = np.array(frame.getData("depth"), dtype="uint16", copy=False)
    ir_map = np.array(frame.getData("ir"), dtype="uint16", copy=False)

    # Create the IR imageq
    new_shape = (int(ir_map.shape[0]),int(ir_map.shape[1]))
    ir_map = ir_map = np.resize(ir_map,new_shape)
    ir_map = distance_scale_ir * ir_map
    ir_map = np.uint8(ir_map)
    ir_map = cv.cvtColor(ir_map, cv.COLOR_GRAY2RGB)
    # print (ir_map.shape)



    # Create the Depth image
    new_shape = (int(depth_map.shape[0]), int(depth_map.shape[1]))
    depth16bits_map = depth_map = np.resize(depth_map, new_shape)
    depth_map = distance_scale * depth_map
    depth_map = np.uint8(depth_map)
    depth_map = cv.applyColorMap(depth_map, cv.COLORMAP_RAINBOW)
    cv.imwrite("DEPTH.png",depth_map)
    
    img_color = cv.absdiff(ir_map, depth_map)
    

    color_image = o3d.geometry.Image(img_color)
    # print (f"color_image is {color_image}")
    depth16bits_image = o3d.geometry.Image(depth_map)
    #75.0 , 3 ---> Best ---> 87.5
    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth16bits_image, 750.0 , 100, True)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth16bits_image, 1.0 , 100, True)

    print (rgbd_image)
    #o3d.io.write_image ("DEPTH.png", rgbd_image.depth )


    plt.subplot (1,2,1)
    plt.title ("Grayscale Image")
    plt.imshow (rgbd_image.color)
    plt.subplot(1,2,2)
    plt.title ("Depth Image")
    plt.imshow(rgbd_image.depth)
    plt.show()

    print ("SHOW")
    # cameraIntrinsics = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cameraIntrinsics )
    o3d.io.write_point_cloud("polygon.ply", pcd)


    # Flip it, otherwise the point cloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.1 , max_nn = 30))
    o3d.visualization.draw_geometries([pcd] )

    # Show the point cloud
    # point_cloud.points = pcd.points
    # point_cloud.colors = pcd.colors
    # if first_time_render_pc:
    #     vis.add_geometry(point_cloud)
    #     first_time_render_pc = 0
    #     # vis.add_geometry(point_cloud) = vis.add_geometry(point_cloud)
    # vis.update_geometry(point_cloud)
    # vis.poll_events()
    # vis.update_renderer()
    
    # if cv.waitKey(1) >= 0:
    #     break
    # ch = cv.waitKey(1)
    # if ch & 0xFF ==ord("q"):
    #     vis.destroy_window()
    
