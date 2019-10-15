import rospy
import sys
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Depth_visulizer():
    def __init__(self):
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/visualized_depth", Image)
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
        visualized_depth = np.full((img.shape[0], img.shape[1], 3), 255)
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


def main(args):
    rospy.init_node("depth_visualizer", anonymous=False)
    depth_visulizer = Depth_visulizer()
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
