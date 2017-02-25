import numpy as np
import cv2
from matplotlib import pyplot as plt

class CloudPoint:
    def __init__(self, point, index_of_2d_origin):
        self.point = point
        self.index_of_2d_origin = index_of_2d_origin
    

def FindCameraMatrices(good, kp1, kp2, matches):
    pts1 = []
    pts2 = []
    #something about if this is good, put it in good
    for m in matches:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # Get the Fundamental Matix
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    print ("F")
    print (F)

    # Initialize Calibration Matrix
    K = np.matrix([(570, 0, 356),
         (0, 562, 300),
         (0, 0, 1)])

    # Calculate Essential Matrix
    E = K.transpose() * F * K
    print("E")
    print (E)

    # Perform Singular Value Decomposition on E
    svd = np.linalg.svd(E, full_matrices=True)
    print ("SVD")
    print (svd)

    #Derive Rotation Matrix and Translation Vector from svd
    W = np.matrix([(0, -1, 0),
         (1, 0, 0),
         (0, 0, 1)])
    R = np.matrix(svd[0] * W * svd[2].transpose())
    print ("R")
    print (R)
    t = np.matrix([(svd[0].item(0,2)),
                   (svd[0].item(1,2)),
                   (svd[0].item(2,2))])
    print("t")
    print(t)

    # Define Camera Matrix for camera 1
    P = np.matrix([(1,0,0,0),
                  (0,1,0,0),
                  (0,0,1,0)])

    # Build Camera Matrix for camera 2
    P1 = np.matrix([(R.item(0,0), R.item(0,1), R.item(0,2), t.item(0)),
                   (R.item(1,0), R.item(1,1), R.item(1,2), t.item(1)),
                   (R.item(2,0), R.item(2,1), R.item(2,2), t.item(2))])

    return F, K, P, P1, pts1, pts2
       

def LinearLSTriangulation(u, P, u1, P1):
    A = np.matrix([(u[0]*P.item(2,0)-P.item(0,0), u[0]*P.item(2,1)-P.item(0,1), u[0]*P.item(2,2)-P.item(0,2)),
                   (u[1]*P.item(2,0)-P.item(1,0), u[1]*P.item(2,1)-P.item(1,1), u[1]*P.item(2,2)-P.item(1,2)),
                   (u1[0]*P1.item(2,0)-P1.item(0,0), u1[0]*P1.item(2,1)-P1.item(0,1), u1[0]*P1.item(2,2)-P1.item(0,2)),
                   (u1[1]*P1.item(2,0)-P1.item(1,0), u1[1]*P1.item(2,1)-P1.item(1,1), u1[1]*P1.item(2,2)-P1.item(1,2))])

    B = np.matrix([(-(u[0]*P.item(2,3)-P.item(0,3))),
                   (-(u[1]*P.item(2,3)-P.item(1,3))),
                   (-(u1[0]*P1.item(2,3)-P1.item(0,3))),
                   (-(u1[1]*P1.item(2,3)-P1.item(1,3)))])
    B = B.T

    #print (A)
    #print (B)

    x = np.linalg.lstsq(A, B)[0]
    return x

def TriangluatePoints(pointcloud, Kinv, pts1, pts2, P, P1):
    for i in range(len(pts1)):
        #print (pts1[i])
        kp = pts1[i]
        u = np.array([kp[0], kp[1], 1.0])
        um = np.array(u * Kinv)[0]
        u = um
        kp1 = pts2[i]
        u1 = np.array([kp1[0], kp1[1], 1.0])
        um1 = np.array(u1 * Kinv)[0] 
        u1 = um1

        X = LinearLSTriangulation(u,P,u1,P1)
        #print ("X")
        #print (X)

        #xPt_img = K * P1 * X

        #print ("X")
        #print (X)
        point = CloudPoint(X, i)
        pointcloud.append(point)

    return



images = []

#img1 = cv2.imread('input/beardshear/Beardshear_Hall.JPG',0)          # queryImage
#img2 = cv2.imread('input/beardshear/Beardshear_Hall_2.JPG',0) # trainImage

#img1 = cv2.imread('input/box/box.png',0)          # queryImage
#img2 = cv2.imread('input/box/box_in_scene.png',0) # trainImage

#images.append(cv2.imread('input/cube/cube_1.JPG',0)) # queryImage
#images.append(cv2.imread('input/cube/cube_2.JPG',0)) # trainImage
#images.append(cv2.imread('input/cube/cube_3.JPG',0)) # trainImage
#images.append(cv2.imread('input/cube/cube_4.JPG',0)) # trainImage

images.append(cv2.imread('input/knome/knome_1.JPG',0)) # queryImage
images.append(cv2.imread('input/knome/knome_2.JPG',0)) # trainImage
images.append(cv2.imread('input/knome/knome_3.JPG',0)) # trainImage
images.append(cv2.imread('input/knome/knome_4.JPG',0)) # trainImage

#img1 = cv2.imread('input/knome_small/knome_1.JPG',0)          # queryImage
#img2 = cv2.imread('input/knome_small/knome_2.JPG',0) # trainImage

#img1 = cv2.imread('input/knome_marker_small/knome_4.JPG',0)          # queryImage
#img2 = cv2.imread('input/knome_marker_small/knome_5.JPG',0) # trainImage

#img1 = cv2.imread('input/rushmore/Mount_Rushmore_1.JPG',0)          # queryImage
#img2 = cv2.imread('input/rushmore/Mount_Rushmore_2.JPG',0) # trainImage


# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(images[0],None)
kp2, des2 = orb.detectAndCompute(images[1],None)

allDes = []
allDes.append(des1)
allDes.append(des2)

allKp = []
allKp.append(kp1)
allKp.append(kp2)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(images[0],kp1,images[1],kp2,matches[:300],None, flags=2)

# Create/Derive all the matrices we need
good = []
F, K, P, P1, pts1, pts2 = FindCameraMatrices(good, kp1, kp2, matches)

#store a list of all the lists of 2d points. Indexed by the order they were added
all2DPoints = []
all2DPoints.append(pts1)
all2DPoints.append(pts2)


# Triangulate Points from the first two images
pointcloud = []
Kinv = K.getI()
TriangluatePoints(pointcloud, Kinv, all2DPoints[0], all2DPoints[0], P, P1)

for i in range(2, len(images)):
    kpi, desi = orb.detectAndCompute(images[i], None)
    bf.match(allDes[i-1], desi)
    allDes.append(desi)
    allKp.append(kpi)
    F, K, P, P1, ptsold, ptsi = FindCameraMatrices(good, allKp[i-1], kpi, matches)
    all2DPoints.append(ptsi)
    TriangluatePoints(pointcloud, Kinv, all2DPoints[i-1], ptsi, P, P1)



file = open("xyz2.xyz", "w")

for i in range(len(pointcloud)):
    print (pointcloud[i])
    file.write(str(pointcloud[i].point[0]).strip('[]').strip() + " " + str(pointcloud[i].point[1]).strip('[]').strip() + " " + str(pointcloud[i].point[2]).strip('[]').strip() + "\n")

file.close()

# display image
#plt.imshow(img3),plt.show()

