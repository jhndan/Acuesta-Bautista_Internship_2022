import sys,os
sys.path.append(os.path.join(os.getcwd(), '.'))
import aditofpython.aditofpython as tof
import cv2 as cv
import mediapipe as mp
import numpy as np
from enum import Enum
import keyboard


#Initialize TOF
WINDOW_NAME_DEPTH = "Display Depth"
WINDOW_NAME_COLOR = "Display Color"
configFile = "aditofpython/tof-viewer_config.json"


class ModesEnum(Enum):
    MODE_NEAR = 0
    MODE_MEDIUM = 1
    MODE_FAR = 2


if __name__ =="__main__":
  

    system = tof.System()

    cameras = []
    status = system.getCameraListAtIp(cameras,"10.42.0.1")

    status = cameras[0].setControl("initialization_config", configFile)

        
    status = cameras[0].initialize()
    

    modes = []
    status = cameras[0].getAvailableModes(modes)


    status = cameras[0].setMode(modes[ModesEnum.MODE_MEDIUM.value])
 

    types = []
    status = cameras[0].getAvailableFrameTypes(types)
   

    status = cameras[0].setFrameType(types[0]) # types[2] is 'mp_pcm' type.
    
 
    
  
    
    status = cameras[0].start()

    
    status = cameras[0].getSensor()


    camDetails = tof.CameraDetails()
    status = cameras[0].getDetails(camDetails)
    if not status:
        print("system.getDetails() failed with status: ", status)

    # Enable noise reduction for better results
    smallSignalThreshold = 100
    cameras[0].setControl("noise_reduction_threshold", str(smallSignalThreshold))

    camera_range = 5000
    bitCount = 5
    frame = tof.Frame()

    max_value_of_IR_pixel = 2 ** bitCount - 1
    distance_scale_ir = 255.0 / max_value_of_IR_pixel
    distance_scale = 255.0 / camera_range

    #Initialize pipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    #SET PIPE
    with mp_pose.Pose(min_detection_confidence = 0.5 ,
                    min_tracking_confidence = 0.5) as pose:
        

        while True :
            #SET TOF TO CAP FRAME BY FRAME
            status = cameras[0].requestFrame(frame)

            if not status:
                print("cameras[0].requestFrame() failed with status: ", status)

            depth_map = np.array(frame.getData("depth"), dtype="uint16", copy=False)
            ir_map = np.array(frame.getData("ir"), dtype="uint16", copy=False)
           
            


            new_shape = (int(depth_map.shape[0]), int(depth_map.shape[1]))
            depth16bits_map = depth_map = np.resize(depth_map, new_shape)
            depth_map = (distance_scale) * depth_map
            depth_map = np.uint8(depth_map)
            depth_map = cv.cvtColor(depth_map, cv.COLOR_GRAY2RGB)

            new_shape = (int(ir_map.shape[0]),int(ir_map.shape[1]))
            ir_map = ir_map = np.resize(ir_map,new_shape)
            # ir_map = (distance_scale_ir*16) * ir_map
            # ir_map = np.uint16(ir_map)
            ir_map = distance_scale * ir_map 
            ir_map = np.uint8(ir_map)
            # ir_map = cv.flip(ir_map, 1)
            ir_map = cv.cvtColor(ir_map, cv.COLOR_GRAY2RGB)

            image = cv.addWeighted (ir_map , 0.4 , depth_map , 0.6 , 0)

            # image = cv.cvtColor(depth_map, cv.COLOR_GRAY2RGB)
            image.flags.writeable = False
            
            results = pose.process(image)

            image.flags.writeable = True
            image = cv.cvtColor (image , cv.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image , results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color = (255,255,255) , thickness = 2 , circle_radius = 2), #dots
                                    mp_drawing.DrawingSpec(color = (255,255,255) , thickness = 2 , circle_radius = 2) #lines
                                    )

            cv.imshow ("Camera" , image)
            cv.waitKey(1)



            if keyboard.is_pressed ('q'):
                cv.destroyAllWindows()
                break
               



