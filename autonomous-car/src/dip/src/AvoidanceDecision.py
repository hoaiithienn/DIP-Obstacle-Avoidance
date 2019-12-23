import cv2
import numpy as np
import rospy

def warn(frame, text):
	cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	rospy.loginfo(text);

def decide(testImage,objects):

	height, width, _ = testImage.shape;
	xMid = int(width / 2);
	yMid = int(height);

	DODGE_ZONE = int(width / 2);
	X_NOT_TURN_LEFT = int(width / 8);
	NOT_TURN_ZONE = int(width / 8);

	xZoneOfInterest = 0;
	yZoneOfInterest = int(1*height / 2);

	xNotTurnLeftStart = xZoneOfInterest + X_NOT_TURN_LEFT;
	xNotTurnLeftEnd = xNotTurnLeftStart + NOT_TURN_ZONE;
	xDodgeZoneStart = xNotTurnLeftEnd;
	xDodgeZoneEnd = xDodgeZoneStart + DODGE_ZONE;
	xNotTurnRightStart = xDodgeZoneEnd;
	xNotTurnRightEnd = xNotTurnRightStart + NOT_TURN_ZONE;

	cv2.line(testImage, (xZoneOfInterest, yZoneOfInterest), (width, yZoneOfInterest), (255, 0, 0), 3);
	cv2.line(testImage, (xNotTurnLeftStart, yZoneOfInterest), (xNotTurnLeftStart, height), (255, 0, 0), 3);
	cv2.line(testImage, (xDodgeZoneStart, yZoneOfInterest), (xDodgeZoneStart, height), (255, 0, 0), 3);
	cv2.line(testImage, (xNotTurnRightStart, yZoneOfInterest), (xNotTurnRightStart, height), (255, 0, 0), 3);
	cv2.line(testImage, (xNotTurnRightEnd, yZoneOfInterest), (xNotTurnRightEnd, height), (255, 0, 0), 3);

	leftSafe = 1;
	rightSafe = 1;
	doDodge = 0;

	for (_,centroid) in objects.items():
		x = centroid[0];
		y = centroid[1];
		# cv2.line(testImage, (xMid, yMid), (x, y), (255, 0, 0),3);
		if (y < yZoneOfInterest):
			# rospy.loginfo("Far");
			warn(testImage, "Far")
		else:
			if (x > xDodgeZoneStart and x < xDodgeZoneEnd and doDodge == 0):
				doDodge = 1;
				if (x < xNotTurnLeftEnd and x > xNotTurnLeftStart and leftSafe == 1):  # x in left CanNotTurn zone
					leftSafe = 0;
				if (x > xNotTurnRightStart and x < xNotTurnRightEnd and rightSafe == 1):  # x in right CanNotTurn zone
					rightSafe = 0;

	if (doDodge):
		if (leftSafe):
			if (x > (xDodgeZoneStart + DODGE_ZONE/2)):
				warn(testImage, "Turn Left")
				# rospy.loginfo("Turn left");
			elif (rightSafe):
				warn(testImage, "Turn Right")
				# rospy.loginfo("Turn right");
			else:
				warn(testImage, "Slow Down")
				# rospy.loginfo("Slow down");
		elif (rightSafe):
			if ( x < (xDodgeZoneStart + DODGE_ZONE/2)):
				warn(testImage, "Turn Right")
				# rospy.loginfo("Turn right");
			else:
				warn(testImage, "Slow Down")
				# rospy.loginfo("Slow down");
		else:
			warn(testImage, "Slow Down")
			# rospy.loginfo("Slowdown");
	else:
		warn(testImage, "Keep Moving")
		# rospy.loginfo("Keep moving");