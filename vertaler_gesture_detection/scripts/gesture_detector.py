#!/usr/bin/env python

#
# Gesture detector
# author: Julio Alvarez
# email: julio.alvarez.d@hotmail.com
# Description: Detect gestures from a given ROS image topic and publish their
# results as a gesture_detection_msg
#

# standard libraries
from __future__ import print_function
import cv2
import sys
import os

# ROS libraries
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError # bridge between cv2 and ROS
from vertaler_gesture_detection_msgs.msg import BoundingBox, ImageBoundingBoxes

# library used to load our trained model:
from darkflow.net.build import TFNet

# Classes and Methods:
#
# RosTensorflow: handles ROS publisher and Subscriber creation, Susbcriber image_callback
# and image conversion from topic to array using CvBridge
#
class RosTensorflow():

    def __init__(self):


        self.pred_pub = rospy.Publisher('od/result', ImageBoundingBoxes, latch=True, queue_size=1)

        self.subs_image_topic = "od/image_to_process"
        rospy.Subscriber(self.subs_image_topic, Image, self.image_callback, queue_size=1)

        self.model = model
        self.bridge = CvBridge()
        self.r = rospy.Rate(10)

    def image_callback(self, _img_msg):
        timestamp_received_img = _img_msg.header.stamp
        frame_id_received_img = _img_msg.header.frame_id
        rospy.loginfo("IMAGE TSTAMP: %s", timestamp_received_img)
        rospy.loginfo("IMAGE FRAME ID: %s", frame_id_received_img)

        cv2_img = self.bridge.imgmsg_to_cv2(_img_msg, "rgb8")

        image_res = ImageBoundingBoxes()
        image_res.header.frame_id = frame_id_received_img
        image_res.header.stamp = timestamp_received_img

        pred = self.model.return_predict(cv2_img)

        if pred:
            boxes = len(pred)

            for box in range(0, boxes):
                bb = BoundingBox()
                result = pred[box]

                label = result.get("label")
                bb.label = label
                confidence = result.get("confidence")
                bb.confidence = confidence
                topleft = result.get("topleft")
                bb.top_left.x = topleft.get("x")
                bb.top_left.y = topleft.get("y")

                bottomright = result.get("bottomright")
                bb.bottom_right.x = bottomright.get("x")
                bb.bottom_right.y = bottomright.get("y")

                image_res.bounding_box.append(bb)

        self.pred_pub.publish(image_res)
        rospy.loginfo("PUBLISHED: %s", image_res.header.stamp)
        self.r.sleep()

    def main(self):
        rospy.spin()

# runNode = Initializes node, create Publisher and Subscriber by calling RosTensorflow
# class and finally exectutes a spin to mantain the script running
def runNode():
    rospy.init_node('predictor')
    predictor = RosTensorflow()
    predictor.main()


if __name__ == '__main__':
    # read model configuration files directory path as an argument:
    model_built_graph_path = rospy.myargv(argv=sys.argv)

    # check if there is only one argument passed at the execution:
    if len(model_built_graph_path) != 2:
        rospy.loginfo("BUILT GRAPH DIRECTORY NOT FOUND")
        sys.exit(1)

    # graph files and labels:
    pb_filename = 'tiny-yolo-voc-22c-eb.pb'
    meta_filename = 'tiny-yolo-voc-22c-eb.meta'
    label_filename = 'labels_eb.txt'

    # Get paths from the plant detection module, this includes cfg, labels, and pb files

    pb_graph_path = os.path.join(model_built_graph_path[1],pb_filename)
    meta_graph_path = os.path.join(model_built_graph_path[1],meta_filename)
    labels_path = os.path.join(model_built_graph_path[1], label_filename)

    # threshold should be 0.5 any value below this range will reduce accuracy of the network
    # gpu 0.65 is the max value of gpu usage on the Tegra, this could change if we switch to another embedded board

    options = {"pbLoad": pb_graph_path, "metaLoad": meta_graph_path, "labels": labels_path,
                    "threshold": 0.5,
                    "gpu": 0.65}

    model = TFNet(options)
    rospy.loginfo("YOLO NETWORK READY")

    runNode()
