import cv2
import numpy as np
import imutils
import dlib
from pyimagesearch.utils import Conf
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import FPS

# load the configuration file
conf = Conf('config/config.json')

#'../project_video.mp4'

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
H = None
W = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=conf["max_disappear"],
	maxDistance=conf["max_distance"])
trackers = []
trackableObjects = {}

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(conf["prototxt_path"],
	conf["model_path"])


cap = cv2.VideoCapture('videos/video1.mp4')
#cap = cv2.VideoCapture(0)


# start the frames per second throughput estimator
fps = FPS().start()

# keep the count of total number of frames
totalFrames = 0


print("[INFO] running detections ...")
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# check if the frame is None, if so, break out of the loop
	if frame is None:
		break

	# resize the frame
	frame = imutils.resize(frame, width=conf["frame_width"])
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	
	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# initialize our list of bounding box rectangles returned by
	# either (1) our object detector or (2) the correlation trackers
	rects = []

		# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames % conf["track_object"] == 0:
		# initialize our new set of object trackers
		trackers = []

		# convert the frame to a blob and pass the blob through the
		# network and obtain the detections
		blob = cv2.dnn.blobFromImage(frame, size=(300, 300),
			ddepth=cv2.CV_8U)
		net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5,
			127.5, 127.5])
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence`
			# is greater than the minimum confidence
			if confidence > conf["confidence"]:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])

				# if the class label is not a car, ignore it
				if CLASSES[idx] != "car":
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
				trackers.append(tracker)


	# otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing
	# throughput
	else:
		# loop over the trackers
		for tracker in trackers:
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
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10)
			, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4,
			(0, 255, 0), -1)



	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


