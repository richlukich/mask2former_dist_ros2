#import rosbag
#from cv_bridge import CvBridge
import cv2
#import image_tools
import numpy as np
import struct
import os
import time

class RoadEdgeDetection():
    def __init__(self, image, mask, depth):
        self.image = image
        self.mask = mask
        self.depth = depth

    def get_xyz(self, x_p,y_p,Z,c_x = 427.2,c_y=240.15,f_x=423.96,f_y=423.96):
        X = (x_p - c_x) * Z / f_x
        Y = (y_p - c_y) * Z / f_y
        dist = np.sqrt(X**2 + Y**2 + Z**2)
        return X, abs(Y), Z, dist

    def grouped(self, iterable, n):
        return zip(*[iter(iterable)]*n)

    def get_max_contour(self, contours):
        areas = [cv2.contourArea(cont) for cont in contours]
        MaxVal = max(areas)
        index_max = areas.index(MaxVal)
        return contours[index_max]

    def delete_edges(self,cnt,mask):
        idx1 = np.where(cnt[:,0] >= mask.shape[1] - 5)
        idx2 = np.where(cnt[:,0] <= 5)
        idx3 = np.where(cnt[:,1] >= mask.shape[0] - 5)
        idx = np.hstack(( idx1,idx2,idx3 )).ravel()
        cnt = np.delete(cnt, idx, 0)
        return cnt
    def get_mean_point(self,cnt, radius = 200):
        #min_y = np.max(cnt[:,1])
        #idx = np.where((cnt[:,1] >= min_y - radius) & (min_y >= cnt[:,1]) )
        #point1 = np.min(cnt[idx][:,0])
        #point2 = np.max(cnt[idx][:,0])
        #return int((point1 + point2)/2)
        left_point = np.min(cnt[:,0])
        right_point  = np.max(cnt[:,0])
    
        return int ((left_point + right_point)/2)

    def get_left_side(self,cnt, mean_point):
        idx_left = cnt[:,0] <= mean_point - 30
        return cnt[idx_left]

    def get_right_side(self,cnt, mean_point):
        idx_right = cnt[:,0] >= mean_point + 30
        return cnt[idx_right]
    
    def find_distances(self):
        k1 = 479 / 720
        k2 = 847 / 1280
        ret, mask = cv2.threshold(self.mask , 0, 255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return []
        if len(contours) > 1:
            max_contour = self.get_max_contour(contours)
        else:
            max_contour = contours[0]
        max_contour = max_contour.reshape(-1, 2)
        max_contour = self.delete_edges(max_contour, mask)
        #M = cv2.moments(max_contour)
        #mean_point = round(M['m10'] / M['m00'])
        mean_point = self.get_mean_point(max_contour)

        left_side = self.get_left_side(max_contour, mean_point)
        right_side = self.get_right_side(max_contour, mean_point)
    
        u_left = list(map(int, left_side[:,0] * k1))
        v_left = list(map(int,left_side[:,1] * k2))
        u_right = list(map(int,right_side[:,0] * k1))
        v_right = list(map(int,right_side[:,1] * k2))

        Z_left = self.depth[v_left,u_left] / 1000
        Z_right = self.depth[v_right,u_right] / 1000

        X_left,Y_left,_,dist_left = self.get_xyz(np.array(u_left),np.array(v_left),Z_left)
        #print (Z_left, Z_right)
        #break
    

        X_right,Y_right,_,dist_right = self.get_xyz(np.array(u_right),np.array(v_right),Z_right)

        
        idx_left, dist_left = np.argmin(dist_left), np.min(dist_left)
        idx_right, dist_right = np.argmin(dist_right), np.min(dist_right)

        u_left, v_left = left_side[idx_left][0],left_side[idx_left][1]
        u_right, v_right = right_side[idx_right][0],right_side[idx_right][1]
       
        return [dist_left, dist_right]