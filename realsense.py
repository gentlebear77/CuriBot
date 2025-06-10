
import pyrealsense2 as rs
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(config)
 
time.sleep(1)
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()
 
# Convert images to numpy arrays
depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

depth_image_resized = cv2.resize(depth_image, (224, 224))
color_image_resized = cv2.resize(color_image, (224, 224))
 
# # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
# depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#
# # Stack both images horizontally
# images = np.hstack((color_image, depth_colormap))
#
# # Show images
# cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('RealSense', images)
# cv2.waitKey(1)
 
cv2.imwrite('realsense_image/color_image_2.png', color_image_resized)
cv2.imwrite('realsense_image/depth_image_1.png', depth_image_resized)
fig, axes = plt.subplots(1, 2)
for ax, im in zip(axes, [color_image_resized, depth_image_resized]):
    ax.imshow(im)
    ax.axis('off')
plt.show()
