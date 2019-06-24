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

# library used to load our trained model:
from darkflow.net.build import TFNet




if __name__ == '__main__':
    print("NODE READY")
