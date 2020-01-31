#!/usr/bin/env python

import cv2
from dynamic_reconfigure.server import Server
from depth_colorizer.cfg import DynamicParamsConfig
import numpy as np
import rospy
import sys

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool, SetBoolResponse


class Depthcolorizer():
    def __init__(self):
        self.srv = Server(DynamicParamsConfig,
                          self.dynamic_reconfigure_callback)
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/colorized_depth", Image, queue_size=1)
        self.pub_mask = rospy.Publisher("/depth_mask", Image, queue_size=1)
        self.pub_kept_mask = rospy.Publisher(
            "/kept_depth_mask", Image, queue_size=1)
        self.use_closing = rospy.get_param(
            '~use_closing', True)
        self.max_distance = rospy.get_param(
            '~max_distance', 1500.)
        self.min_distance = rospy.get_param(
            '~min_distance', 700.)
        self.keep_depth_mask_flag = True
        self.kept_mask = None
        self.subscribe()
        self.service()

    def subscribe(self):
        self.image_sub = rospy.Subscriber("~input",
                                          Image,
                                          self.callback,
                                          queue_size=1)

    def service(self):
        self.service = rospy.Service("/keep_depth_mask",
                                     SetBool, self.keep_depth_mask)

    def keep_depth_mask(self, req):
        self.keep_depth_mask_flag = True
        return SetBoolResponse(True, "call keep_depth_mask")

    def callback(self, msg):
        rospy.loginfo("depthcolorizer is called")
        img = self.bridge.imgmsg_to_cv2(msg, "16UC1")

        # make colorized depth
        colorized_depth = np.full((img.shape[0], img.shape[1], 3), 255,
                                  dtype=np.uint8)
        colorized_depth[:, :, 0] \
            = (img - self.min_distance) * 180. \
            / (self.max_distance - self.min_distance)
        colorized_depth[img < self.min_distance] = 0
        colorized_depth[img > self.max_distance] = 0
        colorized_depth = cv2.cvtColor(colorized_depth.astype(np.uint8),
                                       cv2.COLOR_HSV2RGB)

        # make mask iamge
        mask = cv2.inRange(img, self.min_distance, self.max_distance)
        if self.use_closing:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
        if self.keep_depth_mask_flag:
            self.kept_mask = mask
            self.keep_depth_mask_flag = False

        msg_out = self.bridge.cv2_to_imgmsg(colorized_depth, "rgb8")
        msg_out.header = msg.header

        msg_mask_out = self.bridge.cv2_to_imgmsg(mask, "mono8")
        msg_mask_out.header = msg.header

        msg_kept_mask_out = self.bridge.cv2_to_imgmsg(self.kept_mask, "mono8")
        msg_kept_mask_out.header = msg.header

        self.pub.publish(msg_out)
        self.pub_mask.publish(msg_mask_out)
        self.pub_kept_mask.publish(msg_kept_mask_out)

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
