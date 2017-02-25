import numpy as np
import cv2
from matplotlib import pyplot as plt

#img1 = cv2.imread('Beardshear_Hall.JPG',0)          # queryImage
#img2 = cv2.imread('Beardshear_Hall_2.JPG',0) # trainImage

#img1 = cv2.imread('box.png',0)          # queryImage
#img2 = cv2.imread('box_in_scene.png',0) # trainImage

#img1 = cv2.imread('cube_1.JPG',0)          # queryImage
#img2 = cv2.imread('cube_2.JPG',0) # trainImage

#img1 = cv2.imread('knome_1.JPG',0)          # queryImage
#img2 = cv2.imread('knome_2.JPG',0) # trainImage

img1 = cv2.imread('knome_4.JPG',0)          # queryImage
img2 = cv2.imread('knome_5.JPG',0) # trainImage

#img1 = cv2.imread('Mount_Rushmore_1.JPG',0)          # queryImage
#img2 = cv2.imread('Mount_Rushmore_2.JPG',0) # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:30],None, flags=2)

plt.imshow(img3),plt.show()

#cv.CalibrateCamera2(objectPoints, imagePoints, pointCounts, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags=0)
