import sys
import cv2 as cv
import argparse
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import k4a
import time
from time import perf_counter

inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5
thr = 0.2
WINDOW_NAME = "Display Objects"
WINDOW_NAME_DEPTH = "Display Objects Depth"


if __name__ == "__main__":
	
	# Initialize MobileNetSSD
	parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network ')
	parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt", help='Path to text network file: ' 'MobileNetSSD_deploy.prototxt')
	parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel", help='Path to weights: ' 'MobileNetSSD_deploy.caffemodel')
	args = parser.parse_args()
	try:
		net = cv.dnn.readNetFromCaffe(args.prototxt, args.weights)
	except:
		print("Error: Please give the correct location of the prototxt and caffemodel")
		sys.exit(1)

	swapRB = False
	classNames = {0: 'background',  
                  1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                  5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                  10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                  14: 'motorbike', 15: 'person', 16: 'pottedplant',
                  17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
	
	# Open a device using the static function Device.open().
	device = k4a.Device.open()
	# In order to start capturing frames, need to start the cameras.
    # The start_cameras() function requires a device configuration which
    # specifies the modes in which to put the color and depth cameras.
    # For convenience, the k4a package pre-defines some configurations
    # for common usage of the Azure Kinect device, but the user can
    # modify the values to set the device in their preferred modes.
	device_config = k4a.DEVICE_CONFIG_BGRA32_2160P_WFOV_2X2BINNED_FPS15
	camera_range = 5000
	bitCount = 9
	frame = device.start_cameras(device_config)
	
	max_value_of_IR_pixel = 2 ** bitCount - 1
	distance_scale_ir = 255.0 / max_value_of_IR_pixel
	distance_scale = 255.0 / camera_range
	
	
	prev_frame_time = 0
	new_frame_time = 0

	pre_fps = 0
	smoothing = 0.9

	before_get_data = 0
	after_get_data = 0

	ir_prev_time = 0
	ir_aft_time = 0

	dp_prev_time = 0
	
	while True:
		capture = device.get_capture(-1)
		depth_map = capture.depth.data
		ir_map = capture.ir.data

		# Creation of the IR image
		ir_map = ir_map[0: int(ir_map.shape[0] / 2), :]
		ir_map = distance_scale_ir * ir_map
		ir_map = np.uint8(ir_map)
		ir_map = cv.flip(ir_map, 1)
		ir_map = cv.cvtColor(ir_map, cv.COLOR_GRAY2RGB)
		
		# Creation of the Depth image
		new_shape = (int(depth_map.shape[0] / 2), depth_map.shape[1])
		depth_map = np.resize(depth_map, new_shape)
		depth_map = cv.flip(depth_map, 1)
		distance_map = depth_map
		depth_map = distance_scale * depth_map
		depth_map = np.uint8(depth_map)
		depth_map = cv.applyColorMap(depth_map, cv.COLORMAP_RAINBOW)
		
		new_frame_time=perf_counter()

		acutal = 1 / (new_frame_time-prev_frame_time)
		prev_frame_time = new_frame_time

		fps = (pre_fps * smoothing) + (acutal * (1-smoothing))
		pre_fps = fps

		display_fps = str(int(fps)) + " fps"
		# Combine depth and IR for more accurate results
		result = cv.addWeighted(ir_map, 0.4, depth_map, 0.6, 0)

        # Start the computations for object detection using DNN
		blob = cv.dnn.blobFromImage(result, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), swapRB)

		net.setInput(blob)
		detections = net.forward()

		cols = result.shape[1]
		rows = result.shape[0]

		if cols / float(rows) > WHRatio:
			cropSize = (int(rows * WHRatio), rows)
		else:
			cropSize = (cols, int(cols / WHRatio))

		y1 = int((rows - cropSize[1]) / 2)
		y2 = y1 + cropSize[1]
		x1 = int((cols - cropSize[0]) / 2)
		x2 = x1 + cropSize[0]
		result = result[y1:y2, x1:x2]
		depth_map = depth_map[y1:y2, x1:x2]
		distance_map = distance_map[y1:y2, x1:x2]

		cols = result.shape[1]
		rows = result.shape[0]
		for i in range(detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > thr:
				class_id = int(detections[0, 0, i, 1])
				xLeftBottom = int(detections[0, 0, i, 3] * cols)
				yLeftBottom = int(detections[0, 0, i, 4] * rows)
				xRightTop = int(detections[0, 0, i, 5] * cols)
				yRightTop = int(detections[0, 0, i, 6] * rows)

				cv.rectangle(result, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
							(0, 255, 0))
				cv.rectangle(depth_map, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
							(0, 255, 0))
				center = ((xLeftBottom + xRightTop) * 0.5, (yLeftBottom + yRightTop) * 0.5)

				value_x = int(center[0])
				value_y = int(center[1])
				cv.drawMarker(result, (value_x, value_y), (0, 0, 0), cv.MARKER_CROSS)
				cv.drawMarker(depth_map, (value_x, value_y), (0, 0, 0), cv.MARKER_CROSS)
				if class_id in classNames:
					label_depth = classNames[class_id] + ": " + \
							"{0:.3f}".format(distance_map[value_x, value_y] / 1000.0) + " meters"
					label_conf = "Confidence: " + "{0:.4f}".format(confidence)
					labelSize_depth, baseLine = cv.getTextSize(label_depth, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
					labelSize_conf = cv.getTextSize(label_conf, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

					if labelSize_depth[1] > labelSize_conf[1]:
						labelSize = labelSize_depth
					else:
						labelSize = labelSize_conf
					yLeftBottom = max(yLeftBottom, labelSize[1])
					cv.rectangle(result, (value_x - int(labelSize[0] * 0.5), yLeftBottom),
									(value_x + int(labelSize[0] * 0.5), yLeftBottom + 2 * labelSize[1] + 2 * baseLine),
									(255, 255, 255), cv.FILLED)
					cv.putText(result, label_depth, (value_x - int(labelSize[0] * 0.5), yLeftBottom + labelSize[1]),
								cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
					cv.putText(result, label_conf, (value_x - int(labelSize[0] * 0.5), yLeftBottom + 2 * labelSize[1]
													+ baseLine),
								cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

					cv.rectangle(depth_map, (value_x - int(labelSize[0] * 0.5), yLeftBottom),
									(value_x + int(labelSize[0] * 0.5), yLeftBottom + 2 * labelSize[1] + 2 * baseLine),
									(255, 255, 255), cv.FILLED)
					cv.putText(depth_map, label_depth, (value_x - int(labelSize[0] * 0.5),  yLeftBottom + labelSize[1]),
								cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
					cv.putText(depth_map, label_conf, (value_x - int(labelSize[0] * 0.5), yLeftBottom + 2 * labelSize[1]
														+ baseLine),
								cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


		# Show image with object detection
		cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
		cv.imshow(WINDOW_NAME, result)
		cv.putText(depth_map, display_fps, (7, 20), cv.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
        # Show Depth map
		cv.namedWindow(WINDOW_NAME_DEPTH, cv.WINDOW_AUTOSIZE)
		cv.imshow(WINDOW_NAME_DEPTH, depth_map)
		# Press q key to stop
		if cv.waitKey(1) == ord('q'):  
			break