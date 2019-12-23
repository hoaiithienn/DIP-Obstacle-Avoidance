#!/usr/bin/env python

import roslib
roslib.load_manifest('dip')

import rospy
from std_msgs.msg import String
import cv2
import sys
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
import numpy as np

import imutils
import dlib
import AvoidanceDecision as AD
from pyimagesearch.utils import Conf
from pyimagesearch.centroidtracker import CentroidTracker

from datetime import datetime

class CarController:
	def __init__(self):

		rospy.loginfo("Initializing ROS node ...")
		self.node_name = 'dip_listener'
		rospy.init_node(self.node_name)

		# What the program do during shutdown
		rospy.on_shutdown(self.cleanup)

		# Create the OpenCV display windown for the RGB image	
		# rospy.loginfo("Initializing displays ...")
		# cv2.namedWindow(self.node_name, cv2.WINDOW_NORMAL)
		# cv2.moveWindow(self.node_name, 25, 75)	

		# Create the cv_bridge object
		rospy.loginfo("Creating CVBridge object ...")
		self.bridge = CvBridge()
 
		# load the configuration file
		rospy.loginfo("Loading the configuration file ...")
		self.conf = Conf('config/config.json')

		# instantiate our centroid tracker, then initialize a list to store
		# each of our dlib correlation trackers, followed by a dictionary to
		# map each unique object ID to a TrackableObject
		self.ct = CentroidTracker(maxDisappeared=self.conf["max_disappear"],
			maxDistance=self.conf["max_distance"])
		# self.trackers = []

		# load our serialized model from disk
		rospy.loginfo("Loading model...")
		# initialize the list of class labels MobileNet SSD was trained to
		# detect
		self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]
		self.net = cv2.dnn.readNetFromCaffe(self.conf["prototxt_path"],
			self.conf["model_path"])

		# OTher variables:
		# initialize the frame dimensions (we'll set them as soon as we read
		# the first frame from the video)
		self.H = None
		self.W = None

		# keep the count of total number of frames
		self.totalFrames = 0

		# video recording
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		self.outputvideo = cv2.VideoWriter('pictures/%s.avi'%(current_time),cv2.VideoWriter_fourcc(*'XVID'), 10, (320,240))
		# self.outputvideo = cv2.VideoWriter('pictures/%s.avi'%(current_time),-1, 10, (320,240))

		# Subcribe to the camera image topics and set the appropriate callbacks
		rospy.loginfo("Subscribing to camera topic ....")
		self.image_sub = rospy.Subscriber("dip/camera/rgb/compressed", CompressedImage, self.image_callback, queue_size=1)

		rospy.loginfo("Waiting for image topics ...")


	def image_callback(self, ros_image):
		# Use cv_bridge() to convert the ROS image to OpenCV format
		# rospy.loginfo("Displaying image ...")
		try:
			np_arr = np.fromstring(ros_image.data, np.uint8)
			frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		except CvBridgeError:
		# except:
		 	rospy.loginfo("Error while decoding ...")

		# Convert the image to a Numpy array since most cv2 functions
		# require Numpy arrays.
		frame = np.array(frame, dtype=np.uint8)

		# resize the frame
		frame = imutils.resize(frame, width=self.conf["frame_width"])
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		if self.W is None or self.H is None:
			(H, W) = frame.shape[:2]

		# initialize our list of bounding box rectangles returned by
		# either (1) our object detector or (2) the correlation trackers
		rects = []


		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if self.totalFrames % self.conf["track_object"] == 0:
		# 	# initialize our new set of object trackers
			self.trackers = []

			# convert the frame to a blob and pass the blob through the
			# network and obtain the detections
			blob = cv2.dnn.blobFromImage(frame, size=(300, 300),
				ddepth=cv2.CV_8U)
			self.net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5,
				127.5, 127.5])
			detections = self.net.forward()

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated
				# with the prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by ensuring the `confidence`
				# is greater than the minimum confidence
				if confidence > self.conf["confidence"]:
					# extract the index of the class label from the
					# detections list
					idx = int(detections[0, 0, i, 1])

					# if the class label is not a car, ignore it
					if self.CLASSES[idx] != "car":
						continue

					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")

					# construct a dlib rectangle object from the bounding
					# box coordinates and then start the dlib correlation
					# tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					# add the tracker to our list of trackers so we can
					# utilize it during skip frames
					self.trackers.append(tracker)


		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing
		# throughput
		else:
			# loop over the trackers
			for tracker in self.trackers:
				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects = self.ct.update(rects)

		# Decide
		AD.decide(frame, objects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10)
				, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4,
				(0, 255, 0), -1)

		# Display the image.
		# cv2.imshow(self.node_name, frame)
		# cv2.imwrite("pictures/%s.png"%(rospy.get_time()),frame)
		self.outputvideo.write(frame)

		# if cv2.waitKey(5) & 0xFF == ord('q'):
		# 	rospy.signal_shutdown("User hit q key to quit.")

		# # Process any keyboard commands
		# self.keystroke = cv2.waitKey(5)
		# if 32 <= self.keystroke and self.keystroke < 128:
		# 	cc = chr(self.keystroke).lower()
		# 	if cc == 'q':
		# 		# The user has pressed the q key, so exit
		# 		rospy.signal_shutdown("User hit q key to quit.")

		# increment the total number of frames processed thus far
		self.totalFrames += 1


	def cleanup(self):
		rospy.loginfo("Shutting down vision node")
		cv2.destroyAllWindows()
		self.outputvideo.release()



def main(args):
	try:
		rospy.loginfo("Starting car controller")
		ic = CarController()
		rospy.spin()
	except KeyboardInterrupt:
		ic.cleanup()

if __name__ == '__main__':
	main(sys.argv)
