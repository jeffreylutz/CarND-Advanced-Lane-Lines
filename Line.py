import numpy as np
# import cv2
# import glob
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from moviepy.editor import VideoFileClip
from collections import deque


class Line:
    def __init__(self, left=True):
        # Was the line found in the previous frame?
        self.found = False

        # Is this the left or right side?
        self.left = left

        # Remember x and y values of lanes in previous frame
        self.X = None
        self.Y = None

        # Store recent x intercepts for averaging across frames
        self.x_int = deque(maxlen=10)
        self.top = deque(maxlen=10)

        # Remember previous x intercept to compare against current one
        self.lastx_int = None
        self.last_top = None

        # Remember radius of curvature
        self.radius = None

        # Store recent polynomial coefficients for averaging across frames
        self.fit0 = deque(maxlen=10)
        self.fit1 = deque(maxlen=10)
        self.fit2 = deque(maxlen=10)
        self.fitx = None
        self.pts = []

        # Count the number of frames
        self.count = 0

    def compute_avg_intercepts(self, x, y):
        fit = np.polyfit(y, x, 2)
        int, top = self.get_intercepts(self, fit)
        self.x_int.append(int)
        self.top.append(top)

    def search(self, combined_binary):
        x, y = np.nonzero(np.transpose(combined_binary))
        # Searchfor lane pixels around previous polynomial
        if self.found == True:
            x, y, self.found = self.found_search(x, y)
        else:
            x, y, self.found = self.blind_search(x, y, combined_binary)
        # Calc poly fit based on detected pixels
        fit = np.polyfit(y, x, 2)

        # Compute average x intercepter and top and set lastx_int and last_top
        # averaged across n frames
        x_int, top = self.get_intercepts(self, fit)
        self.x_int.append(x_int)
        self.top.append(top)
        self.lastx_int = np.mean(self.x_int)
        self.last_top = np.mean(self.top)

        # Add averaged intercepts to current x and y vals
        x = np.append(x, x_int)
        y = np.append(y, 720)
        x = np.append(x, top)
        y = np.append(y, 0)

        # Sort detected pixels on y values
        self.X, self.Y = self.sort_vals(x, y)

        # Recalculate polynomial with intercepts and average across n frames
        fit = np.polyfit(self.Y, self.X, 2)
        self.fit0.append(fit[0])
        self.fit1.append(fit[1])
        self.fit2.append(fit[2])
        fit = [np.mean(self.fit0), np.mean(self.fit1), np.mean(self.fit2)]

        # Fit polynomial to detected pixels
        self.fitx = fit[0] * y ** 2 + fit[1] * y + fit[2]

        # Compute radius of curvature for each lane in meters
        if self.count % 3 == 0:
            self.radius = self.radius_of_curvature(x, y)

        return self.x_int

    def found_search(self, x, y):
        '''
        This function is applied when the lane lines have been detected in the previous frame.
        It uses a sliding window to search for lane pixels in close proximity (+/- 25 pixels in the x direction)
        around the previous detected polynomial. 
        '''
        xvals = []
        yvals = []
        if self.found == True:
            i = 720
            j = 630
            while j >= 0:
                yval = np.mean([i, j])
                xval = (np.mean(self.fit0)) * yval ** 2 + (np.mean(self.fit1)) * yval + (np.mean(self.fit2))
                x_idx = np.where((((xval - 25) < x) & (x < (xval + 25)) & ((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    np.append(xvals, x_window)
                    np.append(yvals, y_window)
                i -= 90
                j -= 90
        if np.sum(xvals) == 0:
            self.found = False  # If no lane pixels were detected then perform blind search
        xvals = np.array(xvals).astype(np.float32)
        yvals = np.array(yvals).astype(np.float32)
        return xvals, yvals, self.found

    def blind_search(self, x, y, image):
        '''
        This function is applied in the first few frames and/or if the lane was not successfully detected
        in the previous frame. It uses a slinding window approach to detect peaks in a histogram of the
        binary thresholded image. Pixels in close proimity to the detected peaks are considered to belong
        to the lane lines.
        '''
        xvals = []
        yvals = []
        if self.found == False:
            i = 720
            j = 630
            while j >= 0:
                histogram = np.sum(image[j:i, :], axis=0)
                if self.left == False:
                    peak = np.argmax(histogram[640:]) + 640
                else:
                    peak = np.argmax(histogram[:640])
                x_idx = np.where((((peak - 25) < x) & (x < (peak + 25)) & ((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    xvals.extend(x_window)
                    yvals.extend(y_window)
                i -= 90
                j -= 90
        if np.sum(xvals) > 0:
            self.found = True
        else:
            yvals = self.Y
            xvals = self.X
        xvals = np.array(xvals).astype(np.float32)
        yvals = np.array(yvals).astype(np.float32)
        return xvals, yvals, self.found

    def radius_of_curvature(self, xvals, yvals):
        ym_per_pix = 30. / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
        fit_cr = np.polyfit(yvals * ym_per_pix, xvals * xm_per_pix, 2)
        curverad = ((1 + (2 * fit_cr[0] * np.max(yvals) + fit_cr[1]) ** 2) ** 1.5) \
                   / np.absolute(2 * fit_cr[0])
        return curverad

    def sort_vals(self, xvals, yvals):
        sorted_index = np.argsort(yvals)
        sorted_yvals = yvals[sorted_index]
        sorted_xvals = xvals[sorted_index]
        return sorted_xvals, sorted_yvals

    def get_intercepts(self, polynomial):
        bottom = polynomial[0] * 720 ** 2 + polynomial[1] * 720 + polynomial[2]
        top = polynomial[0] * 0 ** 2 + polynomial[1] * 0 + polynomial[2]
        return bottom, top
