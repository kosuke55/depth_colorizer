#!/usr/bin/env python

from dynamic_reconfigure.server import Server
from depth_colorizer.cfg import DynamicParamsConfig
import rospy
import sys
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Depthcolorizer():
    def __init__(self):
        self.srv = Server(DynamicParamsConfig,
                          self.dynamic_reconfigure_callback)
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/visualized_depth", Image, queue_size=1)
        self.INPUT_IMAGE = rospy.get_param(
            '~input_image', "/head_mount_kinect/hd/image_depth_rect_desktop")
        self.max_distance = rospy.get_param(
            '~max_distance', 1500.)
        self.min_distance = rospy.get_param(
            '~min_distance', 700.)
        self.subscribe()

    def subscribe(self):
        self.image_sub = rospy.Subscriber(self.INPUT_IMAGE,
                                          Image,
                                          self.callback)

    def callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        visualized_depth = np.full((img.shape[0], img.shape[1], 3), 255,
                                   dtype=np.uint8)
        visualized_depth[:, :, 0] \
            = (img - self.min_distance) * 180. \
            / (self.max_distance - self.min_distance)
        visualized_depth[img < self.min_distance] = 0
        visualized_depth[img > self.max_distance] = 0
        visualized_depth = cv2.cvtColor(visualized_depth.astype(np.uint8),
                                        cv2.COLOR_HSV2RGB)
        msg_out = self.bridge.cv2_to_imgmsg(visualized_depth, "rgb8")
        msg_out.header = msg.header
        self.pub.publish(msg_out)

    def dynamic_reconfigure_callback(self, config, level):
        self.min_distance = config["min_distance"]
        self.max_distance = config["max_distance"]
        return config

def main(args):
    rospy.init_node("depthcolorizer", anonymous=False)
    depthcolorizer = Depthcolorizer()
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
