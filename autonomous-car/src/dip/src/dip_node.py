#!/usr/bin/env python

from __future__ import print_function

import roslib
roslib.load_manifest('dip')

import rospy
from std_msgs.msg import String
import cv2 as cv
import sys
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
import numpy as np

class CarController:
	def __init__(self):

		rospy.loginfo("[INFO]: Initializing ROS node ...")
		self.node_name = 'dip_listener'
		rospy.init_node(self.node_name)

		# What the program do during shutdown
		rospy.on_shutdown(self.cleanup)

		# Create the OpenCV display windown for the RGB image
		rospy.loginfo(cv.__version__)		
		rospy.loginfo("[INFO]: Initializing displays ...")
		self.cv_window_name = self.node_name
		cv.namedWindow(self.cv_window_name, cv.WINDOW_NORMAL)
		cv.moveWindow(self.cv_window_name, 25, 75)	

		# Create the cv_bridge object
		rospy.loginfo("[INFO]: Creating CVBridge object ...")
		self.bridge = CvBridge()

		# Subcribe to the camera image topics and set the appropriate callbacks
		rospy.loginfo("[INFO]: Subscribing to camera topic ....")
		self.image_sub = rospy.Subscriber("dip/camera/rgb/compressed", CompressedImage, self.image_callback, queue_size=1)

		rospy.loginfo("[INFO]: Waiting for image topics ...")

	# def listener():

	# 	# In ROS, nodes are uniquely named. If two nodes with the same
	# 	# name are launched, the previous one is kicked off. The
	# 	# anonymous=True flag means that rospy will choose a unique
	# 	# name for our 'listener' node so that multiple listeners can
	# 	# run simultaneously.
	# 	rospy.init_node('listener', anonymous=True)

	# 	rospy.Subscriber('chatter', String, callback)

	# 	# spin() simply keeps python from exiting until this node is stopped
	# 	rospy.spin()

	def image_callback(self, ros_image):
		# Use cv_bridge() to convert the ROS image to OpenCV format
		rospy.loginfo("[INFO]: Displaying image ...")
		try:
			np_arr = np.fromstring(ros_image.data, np.uint8)
			frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)

			# frame = self.bridge.imgmsg_to_cv(ros_image, "bgr8")
		except CvBridgeError:
			rospy.loginfo(e)

		# Convert the image to a Numpy array since most cv2 functions
		# require Numpy arrays.
		frame = np.array(frame, dtype=np.uint8)

		# Process the frame using the process_image() function
		# display_image = self.process_image(frame)

		# Display the image.
		cv.imshow(self.node_name, frame)

		# Process any keyboard commands
		self.keystroke = cv.waitKey(5)
		if 32 <= self.keystroke and self.keystroke < 128:
			cc = chr(self.keystroke).lower()
			if cc == 'q':
				# The user has pressed the q key, so exit
				rospy.signal_shutdown("User hit q key to quit.")

	def cleanup(self):
		rospy.loginfo("Shutting down vision node")
		cv.destroyAllWindows()


def main(args):
	try:
		rospy.loginfo("Starting car controller")
		ic = CarController()
		rospy.spin()
	except KeyboardInterrupt:
		rospy.loginfo("[INFO]: Shutting down vision node.")
		cv.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
