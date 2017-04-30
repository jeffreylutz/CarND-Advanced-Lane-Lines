import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from moviepy.editor import VideoFileClip


overhead = cv2.imread('test_images/test7.jpg')
#h,l = cv2.cvtColor(overhead, cv2.COLOR_BGR2HLS)
(h,l,s) = cv2.cvtColor(overhead, cv2.COLOR_BGR2HLS)[:,:,:]

plt.imshow(cv2.cvtColor(overhead,cv2.COLOR_BGR2RGB) )
plt.show()
plt.imshow(hls[0])
plt.show()
exit(1)

(h,l,s) = cv2.cvtColor(overhead, cv2.COLOR_BGR2HLS)[:,:,:]
l_channel = cv2.cvtColor(overhead, cv2.COLOR_BGR2LUV)[:, :, 0]
b_channel = cv2.cvtColor(overhead, cv2.COLOR_BGR2Lab)[:, :, 2]

# Threshold color channel
s_thresh_min = 150
s_thresh_max = 255
s_binary = np.zeros_like(s)
s_binary[(s >= s_thresh_min) & (s <= s_thresh_max)] = 1

b_thresh_min = 155
b_thresh_max = 190
b_binary = np.zeros_like(b_channel)
b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

l_thresh_min = 210
l_thresh_max = 255
l_binary = np.zeros_like(l_channel)
l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

combined_binary = np.zeros_like(s_binary)
combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

min = 150
max = 255

min = 200
max = 255
while True:
    print('min',min)
    # min = int(input('Enter min: '))
    # max = int(input('Enter max: '))
    s_binary = np.zeros_like(s)
    s_binary[(s >= 210) & (s <= 255)] = 1
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= 170) & (b_channel <= 255)] = 1
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 210) & (l_channel <= 255)] = 1

    # cv2.imshow('Hi',s_binary)
    # cv2.imshow('Hi',combined_binary)
    plt.imshow(s_binary)
    plt.show()
    plt.imshow(b_binary)
    plt.show()
    plt.imshow(l_binary)
    plt.show()
    # if cv2.waitKey(1000) == 27 or min > max:
    #     break
    # min = min + 1

cv2.destroyAllWindows()