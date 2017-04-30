# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

f, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3, figsize=(5,5))

# load the image
image = cv2.imread('test_images/test7.jpg')

hsl = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

hsl_h = hsl[:,:,0]
hsl_s = hsl[:,:,1]
hsl_l = hsl[:,:,2]
hsv_h = hsv[:,:,0]
hsv_s = hsv[:,:,1]
hsv_v = hsv[:,:,2]

# ax1.imshow(hsl_h)
# ax2.imshow(hsl_s)
# ax3.imshow(hsl_l)
# ax4.imshow(hsv_h)
# ax5.imshow(hsv_s)
# ax6.imshow(hsv_v)
# plt.show()

lo = np.array([50,150,50])
hi = np.array([255,255,180])
while(1):
	mask = cv2.inRange(hsv, lo, hi)
	res = cv2.bitwise_and(image, image, mask=mask)
	cv2.imshow('d',image)
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
