#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import cv2 as cv
import sys
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class CarController:
	def __init__(self):
		self.node_name = 'dip'
		rospy.init_node(self.node_name)

		# What the program do during shutdown
		rospy.on_shutdown(self.cleanup())

		# Create the OpenCV display windown for the RGB image		
		self.cv_window_name = self.node_name
		cv.NamedWindow(self.cv_window_name, cv.CV_WINDOW_NORMAL)
		cv.MoveWindow(self.cv_window_name, 25, 75)	

		# Create the cv_bridge object
		self.bridge = CvBridge()

		# Subcribe to the camera image topics and set the appropriate callbacks
		self.image_sub = rospy.Subscriber("dip/camera/rgb", Image, self.image_callback)

		rospy.loginfo("Waiting for image topics...")

	def image_callback(self, ros_image):
		# Use cv_bridge() to convert the ROS image to OpenCV format
		try:
			frame = self.bridge.imgmsg_to_cv(ros_image, "bgr8")
		except CvBridgeError, e:
			print e

		# Convert the image to a Numpy array since most cv2 functions
		# require Numpy arrays.
		frame = np.array(frame, dtype=np.uint8)

		# Process the frame using the process_image() function
		# display_image = self.process_image(frame)

		# Display the image.
		cv2.imshow(self.node_name, frame)

		# Process any keyboard commands
		self.keystroke = cv.WaitKey(5)
		if 32 <= self.keystroke and self.keystroke < 128:
			cc = chr(self.keystroke).lower()
			if cc == 'q':
				# The user has pressed the q key, so exit
				rospy.signal_shutdown("User hit q key to quit.")

	def cleanup(self):
		print "Shutting down vision node."
		cv.DestroyAllWindows()


def main(args):
	try:
		ic = CarController()
		rospy.spin()
	except KeyboardInterrupt:
		print "Shutting down vision node."
		cv.DestroyAllWindows()

if __name__ == 'main':
	main(sys.argv)