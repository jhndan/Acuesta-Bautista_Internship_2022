from stat import S_IMODE, ST_MODE
import sys,os
sys.path.append(os.path.join(os.getcwd(), '.'))
import aditofpython.aditofpython as tof
import numpy as np
import cv2 as cv
from enum import Enum 
from time import perf_counter
import keyboard
import open3d as o3d

WINDOW_NAME_DEPTH = "Display Depth"
WINDOW_NAME_COLOR = "Display Color"
configFile = "aditofpython/tof-viewer_config.json"


class ModesEnum(Enum):
    MODE_NEAR = 0
    MODE_MEDIUM = 1
    MODE_FAR = 2


def select_modes (frame  , distance_scale_ir , distance_scale , cameraIntrinsics):
    print ("DIFFERENT MODES \n [1]:Show Fps \n [2]:Show Depth Map \n [3]:Active Brightness \n [4]:PCD points \n [0]:Exit \n")
    s_mode= abs(eval(input("What Mode Do you want : ")))
    if s_mode == 0 :
        sys.exit()
    
    
    frame = frame
    distance_scale_ir = distance_scale_ir
    distance_scale = distance_scale
    cameraIntrinsics = cameraIntrinsics
    

    return s_mode , frame , distance_scale_ir , distance_scale , cameraIntrinsics

#PRE PROCESS OF DEPTH
def pre_process_depth_map (depth_map , distance_scale ):

    new_shape = (int(depth_map.shape[0]), int(depth_map.shape[1]))
    depth_map = depth_map = np.resize(depth_map, new_shape)
    depth_map = distance_scale * depth_map
    depth_map = np.uint8(depth_map)
    depth_map = cv.applyColorMap(depth_map, cv.COLORMAP_RAINBOW)
    


    return depth_map 

#PRE PROCESS OF IR
def pre_process_ir_map (ir_map , distance_scale_ir):
    new_shape = (int(ir_map.shape[0]),int(ir_map.shape[1]))
    ir_map = ir_map = np.resize(ir_map,new_shape)
    ir_map = distance_scale_ir * ir_map 
    ir_map = np.uint8(ir_map)
    #ir_map = cv.flip(ir_map, 1)
    ir_map = cv.cvtColor(ir_map, cv.COLOR_GRAY2RGB)

    return ir_map

#GETTING FPS
def get_fps (prev_time , new_time , prev_fps  ,smoothing):
    actual_fps = 1 / (new_time - prev_time)
    prev_time = new_time
    actual_fps = (prev_fps * smoothing) + (actual_fps * (1 - smoothing))
    prev_fps = actual_fps
    return actual_fps , prev_time , prev_fps

#TRANSFORMING IMAGE
def transform_image(np_image):
    return o3d.geometry.Image(np_image)








if __name__ =="__main__":
    
    
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
        print("system.getAvailableModes() failed with status: ", status)

    status = cameras[0].setMode(modes[ModesEnum.MODE_MEDIUM.value])
    if not status:
        print("cameras[0].setMode() failed with status: ", status)

    print (f"\nstatus {status} setMode\n")

    types = []
    status = cameras[0].getAvailableFrameTypes(types)
    if not status:
        print("system.getAvailableFrameTypes() failed with status: ", status)

    print (f"\nstatus {status}getavailableframetype {types} \n")

    status = cameras[0].setFrameType(types[0]) # types[2] is 'mp_pcm' type.
    
    print (f"\nstatus {status} the error is{cameras[0].setFrameType(types[0])}\n ")
    
    if not status:
        print("cameras[0].setFrameType() failed with status:", status)
    
    status = cameras[0].start()
    if not status:
        print("cameras[0].start() failed with status:", status)
    
    status = cameras[0].getSensor()

    print (f"\nSDASDASDASD{status}SADASDASDASD\n")

    camDetails = tof.CameraDetails()
    status = cameras[0].getDetails(camDetails)
    if not status:
        print("system.getDetails() failed with status: ", status)

    # Enable noise reduction for better results
    smallSignalThreshold = 100
    cameras[0].setControl("noise_reduction_threshold", str(smallSignalThreshold))

    camera_range = 1000
    bitCount = 9
    frame = tof.Frame()
    status = cameras[0].requestFrame(frame)
    frameDataDetails = tof.FrameDataDetails()
    status = frame.getDataDetails("depth", frameDataDetails)
    width = frameDataDetails.width
    height = frameDataDetails.height


    max_value_of_IR_pixel = 2 ** bitCount - 1
    distance_scale_ir = 255.0 / max_value_of_IR_pixel
    distance_scale = 255.0 / camera_range

    # Get intrinsic parameters from camera
    intrinsicParameters = camDetails.intrinsics
    fx = intrinsicParameters.fx
    
    fy = intrinsicParameters.fy
    
    cx = intrinsicParameters.cx

    cy = intrinsicParameters.cy
    
    cameraIntrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

   
    
    
   
   
        
    

    def initialize (smode , frame  , distance_scale_ir , distance_scale , cameraIntrinsics ):

        smode = smode
        frame = frame
        distance_scale_ir = distance_scale_ir
        distance_scale = distance_scale
        cameraIntrinsics = cameraIntrinsics
        time_initialize = 0
        #Smooting
        smoothing  = 0.9
        #FPS OF RAW DEPTH FRAME
        prev_fraw_depth_time = 0
        new_fraw_depth_time = 0
        prev_fraw_depth_fps = 0
        

        #FPS OF RAW IR FRAME
        prev_fraw_ir_time = 0
        new_fraw_ir_time = 0
        real_fraw_ir_time = 0
        prev_fraw_ir_fps = 0

        #OPEN3d VIS
        vis = o3d.visualization.Visualizer()
        first_time_render_pc = 1
        point_cloud = o3d.geometry.PointCloud()
     
    
        while True:
            # Capture frame-by-frame
            status = cameras[0].requestFrame(frame)
            if not status:
                print("cameras[0].requestFrame() failed with status: ", status)
            time_initialize = perf_counter()

            #Frame of depth
            depth_map = np.array(frame.getData("depth"), dtype="uint16", copy=False)


            new_fraw_depth_time = perf_counter()
            current_fraw_depth_fps , prev_fraw_depth_time , prev_fraw_depth_fps = get_fps (prev_fraw_depth_time ,
                                                                                            new_fraw_depth_time ,
                                                                                            prev_fraw_depth_fps ,  
                                                                                            smoothing )

            # Frame of I.R
            ir_map = np.array(frame.getData("ir"), dtype="uint16", copy=False)
            new_fraw_ir_time = perf_counter()
            real_fraw_ir_time = (new_fraw_ir_time - new_fraw_depth_time) + time_initialize
            current_fraw_ir_fps , prev_fraw_ir_time , prev_fraw_ir_fps = get_fps(prev_fraw_ir_time ,
                                                                                real_fraw_ir_time ,
                                                                                prev_fraw_ir_fps ,
                                                                                smoothing)


            depth_map  = pre_process_depth_map(depth_map , distance_scale)
            # depth16bits_map = pre_process_depth_map(depth_map , distance_scale)[1]
            ir_map = pre_process_ir_map(ir_map, distance_scale_ir)

            #pcd proccess
            # img_color =  cv.addWeighted(ir_map, 0.4, depth_map, 0.6, 0)
            img_color = cv.absdiff (ir_map , depth_map)
            color_image = o3d.geometry.Image(img_color)
            depth16bits_image = o3d.geometry.Image(depth_map)

            # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth16bits_image, 15000.0, 100.0, False)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth16bits_image, 750.0, 100, True)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cameraIntrinsics)

            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 10 , max_nn = 15))








            #PRINTED RAW FRAME RATE
            if smode == 1:
                print (f"Raw_Depth_FPS : {current_fraw_depth_fps} ------ Raw_IR_fps : {current_fraw_ir_fps}")
                if keyboard.is_pressed('Esc') :
                    print (f"EXIT CURRENT MODE")
                    run(frame  , distance_scale_ir , distance_scale , cameraIntrinsics)
                    break
            

            #DISPLAY DEPTH FRAME W/FPS
            elif smode == 2:

            # result = cv.addWeighted(raw_map, 0.4, depth_map, 0.6, 0)
                cv.putText(depth_map, f"{int(current_fraw_depth_fps)} FPS", (7, 75), cv.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 2,cv.LINE_AA)
                cv.namedWindow(WINDOW_NAME_DEPTH, cv.WINDOW_AUTOSIZE)
                cv.imshow(WINDOW_NAME_DEPTH, depth_map)
                cv.waitKey(1)
                

                if keyboard.is_pressed('Esc'):
                    cv.destroyAllWindows()
                    print (f"EXIT CURRENT MODE")
                    run(frame  , distance_scale_ir , distance_scale , cameraIntrinsics)
                    break
                


            #DISPLAY I.R FRAME W/ FPS
            elif smode == 3:

            
                cv.putText(ir_map, f"{int(current_fraw_ir_fps)} FPS", (7, 75), cv.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 2,cv.LINE_AA)
                cv.namedWindow(WINDOW_NAME_COLOR, cv.WINDOW_AUTOSIZE)
                cv.imshow(WINDOW_NAME_COLOR, ir_map)
                cv.waitKey(1)

                if keyboard.is_pressed('Esc'):
                    print (f"EXIT CURRENT MODE")
                    cv.destroyAllWindows()
                    run(frame  , distance_scale_ir , distance_scale , cameraIntrinsics)
                    break
            
            elif smode == 4 :

                # Show the point cloud
                vis.create_window("PointCloud", 1200, 1200)
                pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 10 , max_nn = 15))
                point_cloud.points = pcd.points
                point_cloud.colors = pcd.colors
                if first_time_render_pc:
                    vis.add_geometry(point_cloud)
                    first_time_render_pc = 0
                vis.update_geometry(point_cloud)
                vis.poll_events()
                vis.update_renderer()
                #SHOW MO UNG RESULT NG DALAWANG IMAGE PARA MAKITA UNG SA PCD
                if keyboard.is_pressed('Esc') :
                    print (f"EXIT CURRENT MODE")
                    vis.destroy_window()
                    run(frame  , distance_scale_ir , distance_scale , cameraIntrinsics)
                    break


            if cv.waitKey(1) >= 0:
                print (f"EXIT CURRENT MODE")
                break
    def run (frame , distance_scale_ir , distance_scale , cameraIntrinsics):
        
        smode ,  frame , distance_scale_ir , distance_scale , cameraIntrinsics = select_modes(frame , distance_scale_ir 
                                                                                        , distance_scale , cameraIntrinsics)

        if smode <= 4  :
            initialize(smode ,  frame , distance_scale_ir , distance_scale , cameraIntrinsics)
        else :
            sys.exit()
    
    run(frame , distance_scale_ir , distance_scale , cameraIntrinsics)








        




