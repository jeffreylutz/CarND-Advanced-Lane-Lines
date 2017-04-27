import numpy as np
import cv2
# import glob
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from moviepy.editor import VideoFileClip
from collections import deque
from Line import Line


class ImageProcessor:
    def __init__(self, cal_image):
        # Was the line found in the previous frame?
        self.Left = Line(left=True)
        self.Right = Line(left=False)
        self.mtx, self.dist = self.calibrate(self, cal_image)

    def calibrate(self, cal_image):
        image_size = (cal_image.shape[0], cal_image.shape[1])
        # Calibrate camera and undistort image
        objpoints = []
        imgpoints = []
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
        return mtx, dist

    def process(self, image):
        undist = self.undistort(self, image)
        warped = self.warp(self, undist)
        combined_binary = self.binaryThreshold(self, warped)

        # Identify all non-zero values in the image
        leftx_int = self.Left.search(combined_binary)
        rightx_int = self.Right.search(combined_binary)
        # Calc position of car relative to the center of the lane
        lane_center = 640
        m_per_px = 3.7 / 700.
        position = (leftx_int + rightx_int) / 2
        # + == left of center,  - == right of center
        off_center = (lane_center - position) * m_per_px

        Minv = cv2.getPer


    def undistort(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    def warp(self, image):
        image_size = (image.shape[0], image.shape[1])
        # Perform perspective transform
        offset = 0
        src = np.float32([[490, 482], [810, 482],
                          [1250, 720], [0, 720]])
        dst = np.float32([[0, 0], [1280, 0],
                          [1250, 720], [40, 720]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, image_size)

    def binaryThreshold(self, image):
        # Generate binary thresholded images
        b_channel = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)[:, :, 2]
        l_channel = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)[:, :, 0]

        # Set the upper and lower thresholds for the b channel
        b_thresh_min = 145
        b_thresh_max = 200
        b_binary = np.zeros_like(b_channel)
        b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

        # Set the upper and lower thresholds for the l channel
        l_thresh_min = 215
        l_thresh_max = 255
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

        combined_binary = np.zeros_like(b_binary)
        combined_binary[(l_binary == 1) | (b_binary == 1)] = 1
        return combined_binary
