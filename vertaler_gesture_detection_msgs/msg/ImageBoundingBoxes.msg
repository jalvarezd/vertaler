#
# Image Bounding Boxes
# Description: Each Image could contain multiple bounding boxes, this message take all
# the object detected on a single image and pack them into a single message
#
# Attributes:
# header = generic information from ROS
# bounding_box = Array of variable lenght containing detected bounding boxes from the
# detection

Header header
BoundingBox[] bounding_box
