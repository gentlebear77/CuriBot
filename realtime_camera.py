import pyrealsense2 as rs
import numpy as np
import cv2

# Create a pipeline object
pipeline = rs.pipeline()

# Configure the pipeline to stream color and depth data
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a frame from the camera
        frames = pipeline.wait_for_frames()
        
        # Get the color and depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert the depth frame to a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert the color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Normalize the depth image to be between 0 and 255 for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack the color and depth images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Display the images
        cv2.imshow('RealSense', images)

        # Wait for the user to press 'q' to exit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
