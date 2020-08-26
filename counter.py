from centroidtracker import CentroidTracker, TrackableObject
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from scipy.spatial import distance as dist

# dict containing all configurable values
inputs = {
	'prototxt': 'detector/MobileNetSSD_deploy.prototxt',
	'model': 'detector/MobileNetSSD_deploy.caffemodel',
	'input': 'videos/sample-video.avi',
	'confidence': 0.7,
	'skip_frames': 25
}

# MANDATORY - mobilenetssd class labels
CLASSES = [
	"background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"
]

# instantiate 1 - Object Detector, 2 - Video Source, 3 - Centroid Tracker
net = cv2.dnn.readNetFromCaffe(inputs["prototxt"], inputs["model"])
vs = cv2.VideoCapture(inputs["input"])
ct = CentroidTracker(maxDisappeared=30, maxDistance=50)

trackers = []
trackableObjects = {}

# counters
totalDown = 0
totalUp = 0
count = 0
totalFrames = 0

durations = []
totalFrameArray = []
fps = []
countArray = []

W = None
H = None

# fps counter
begin = time.time()

while True:
	start = time.time()
	frame = vs.read()
	frame = frame[1]

	# break out of loop when last frame reached
	if inputs["input"] is not None and frame is None:
		break

	# resize the frame to be 480px wide to reduce the data the program has to process
	frame = cv2.resize(frame, (480, 360))
	(H, W) = frame.shape[:2]

	rects = []

	# run Object Detector every SkipFrames to reduce power usage
	if totalFrames % inputs["skip_frames"] == 0:
		trackers = []

		# convert the frame to a blob and pass the blob through the neural network to get detections
		# assuming a mean subtraction value of 127.5 (255/2) and using 1/Mean Subtraction Value to calculate the scale
		blob = cv2.dnn.blobFromImage(frame, 1/127.5, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			# run if confidence is above threshold
			if confidence > inputs["confidence"]:
				# get label
				index = int(detections[0, 0, i, 1])

				# fliter label
				if CLASSES[index] != "person":
					continue
				# (x, y) co-ordinates
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				startX, startY, endX, endY = box.astype("int")

				startX = startX.item()
				startY = startY.item()
				endX = endX.item()
				endY = endY.item()

				boxes = [startX, startY, endX - startX, endY - startY]
				
				# make sure bounding box coordinates do not exceed frame boundaries
				for i in range(len(boxes)):
					if boxes[i] >= W:
						boxes[i] = W
					elif boxes[i] >= H:
						boxes[i] = H
					elif boxes[i] < 0:
						boxes[i] = 0

				tracker = cv2.TrackerKCF_create()
				success = tracker.init(frame, tuple(boxes))

				trackers.append(tracker)

	# use object tracking for all other frames
	else:
		for tracker in trackers:
			success, bbox = tracker.update(frame)
			if success:
				startX = int(bbox[0])
				startY = int(bbox[1])
				endX = int(bbox[0] + bbox[2])
				endY = int(bbox[1] + bbox[3])

			rects.append((startX, startY, endX, endY))
			cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 3)

	cv2.line(frame, (0, int(H/2)), (W, int(H/2)), (255, 0, 0), 3)

	# run centroid tracking algorithm
	objects = ct.update(rects)

	for (objectID, centroid) in objects.items():
		to = trackableObjects.get(objectID, None)

		if to is None:
			to = TrackableObject(objectID, centroid)

		else:
			# calculate direction by using current y-coordinate with the average y coordinate of all previous positions
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			if not to.counted:
				# if direction is negative then object is going up
				if direction < 0 and centroid[1] < int(H / 2):
					totalUp += 1
					to.counted = True

				# if the direction is positive then object is going down
				elif direction > 0 and centroid[1] > int(H / 2):
					totalDown += 1
					to.counted = True

		trackableObjects[objectID] = to

		text = "ID: {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	cv2.putText(frame, "In: " + str(totalUp), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	cv2.putText(frame, "Out: " + str(totalDown), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	cv2.putText(frame, "Total Count: " + str(totalUp - totalDown), (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

	# resize the frame to a larger size to see the output
	frame = cv2.resize(frame, (1000, 750))
	cv2.imshow("Result", frame)
	key = cv2.waitKey(1) & 0xFF
	totalFrames += 1

	stop = time.time()
	duration = stop - start

	if totalFrames > 1:
		totalFrameArray.append(totalFrames)
		countArray.append(len(trackers))
		durations.append(duration)

	if duration > 0:
		fps.append(1 / duration)

	if key == ord("q"):
		break
end = time.time()
print("FPS: " + str(totalFrames/(end - begin)))

print("Up: " + str(totalUp))
print("Down: " + str(totalDown))
print("Total Count: " + str(totalUp - totalDown))

vs.release()
cv2.destroyAllWindows()

''' Graphs for performance analysis
fig, frameTime = plt.subplots()
fig, countGraph = plt.subplots()

frameTime.plot(totalFrameArray, durations, label='Time taken to process each frame')
frameTime.plot([0, totalFrameArray[-1]], [1/30, 1/30], label='1/30th of a second')
frameTime.plot([0, totalFrameArray[-1]], [1/24, 1/24], label='1/24th of a second')
frameTime.set_xlabel('Duration (frames)')
frameTime.set_ylabel('Time (seconds)')
frameTime.set_title('Processing Time')
frameTime.legend()

countGraph.plot(totalFrameArray, countArray, label='Total Count')
countGraph.set_xlabel('Duration (frames)')
countGraph.set_ylabel('Number of People in the frame')
countGraph.set_yticks(np.arange(0, 6, 1))
plt.show()
'''
