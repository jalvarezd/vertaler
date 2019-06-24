#
# BoundingBox message
# Description: Bounding Box of a single object from the Deep Learning
# gesture detection algorithm
#
# Attributes:
# label = gesture class
# confidence = range between 0-1 which tells how confidence is the inference about that object prediction
# top_left = top left coordinates of the bounding box
# bottom_right = bottom_right coordinates of the bounding box

geometry_msgs/Point top_left
geometry_msgs/Point bottom_right
string label
float32 confidence
