import rclpy
import cv2
import numpy as np
import message_filters

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from semseg_ros2.road_detection import RoadEdgeDetection
from semseg_ros2 import  image_tools
from std_msgs.msg import Int32MultiArray, Float64MultiArray

class DistanceNode(Node):
    def __init__(self):
        super().__init__('distance_node')
        #print ('GOOD GOOD GOOD GOOD GOOD GOOD GOOD GOOD GOOD GOOD GOOD GOOD')
        image_sub = message_filters.Subscriber(self, CompressedImage, 'image')
        segmentation_sub = message_filters.Subscriber(self, Image, 'segmentation')
        #depth_sub = message_filters.Subscriber(self, CompressedImage, 'depth')
        depth_sub = message_filters.Subscriber(self, CompressedImage,'/realsense_back/depth/image_rect_raw/compressedDepth')
        self.ts = message_filters.TimeSynchronizer([image_sub, segmentation_sub,depth_sub], 10)
        self.ts.registerCallback(self.road_edge_detection)

        self.distances = self.create_publisher(Float64MultiArray, 'distances', 10)

        self.br = CvBridge()
    def road_edge_detection(self, image_msg: CompressedImage, segm_msg: Image, depth_msg: CompressedImage):
        image = self.br.compressed_imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        mask = self.br.imgmsg_to_cv2(segm_msg, desired_encoding='mono8')
        mask = np.where(mask==3,mask,0)
        depth = image_tools.it.convert_compressedDepth_to_cv2(depth_msg)
        depth[depth == 0] = 15000
        road_detection = RoadEdgeDetection(image, mask, depth)

        distances = road_detection.find_distances()
        #print ("DISTANCES", distances)
        distances_msg = Float64MultiArray()
        distances_msg.data = distances
        self.distances.publish(distances_msg)
def main(args=None):
    rclpy.init(args=args)

    node = DistanceNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
        
        

