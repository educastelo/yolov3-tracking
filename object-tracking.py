# import the necessary packages
from tracker.centroid import CentroidTracker
from utilities.utils import point_in_polygons, draw_roi
from imutils.video import FPS
import numpy as np
import time
import cv2
import os
import cvzone


def show_fps(imagem, frames_persec):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {:.2f}'.format(frames_persec)
    cv2.putText(imagem, fps_text, (11, 20), font, 2.0, (32, 32, 32), 4, line)
    cv2.putText(imagem, fps_text, (10, 20), font, 2.0, (240, 240, 240), 1, line)
    return imagem


# setting the ROI (polygon) of the frame and loading the video stream
points_polygon = [[[827, 484], [286, 1057], [52, 809], [398, 518], [833, 478]]]
stream = u'rtmp://rtmp...'
vs = cv2.VideoCapture(stream)

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=1, maxDistance=400)
trackers = []
trackableObjects = {}
(H, W) = (None, None)

# set the confidence level
confidenceLevel = 0.1
threshold = 0.3

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["yolov3", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolov3", "yolov3.weights"])
configPath = os.path.sep.join(["yolov3", "yolov3.cfg"])

# load our YOLO object model trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# start the frames per second throughput estimator
time.sleep(2.0)
fps = FPS().start()
fps_ = 0.0
tic = time.time()

totalFrames = 0

writer = None

# loop over the frames from the video stream
while True:
    # read the next frame from the video stream and resize it
    _, frame = vs.read()

    # if the frame dimensions are None, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object model, giving us our bounding boxes
    # and associated probabilities
    start = time.time()
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    net.setInput(blob)
    layerOutputs = net.forward(ln)
    rects = []
    end = time.time()
    print("[INFO] classification time " + str((end - start) * 1000) + "ms")

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidenceLevel:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidenceLevel,
                            threshold)

    rects = []
    classDetected_list = []
    confDegree_list = []

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if LABELS[classIDs[i]] not in ["person", "car", "motorbike", "bus", "bicycle", "truck"]:
                continue

            # Check if the centroid is inside the polygon
            cX = x + w / 2
            cY = y + h / 2
            test_polygon = point_in_polygons((cX, cY), points_polygon)
            if not test_polygon:
                continue

            # mostra retangulo com corner customizado
            cvzone.cornerRect(frame, (x, y, w, h), l=10, t=4)
            cvzone.putTextRect(frame,
                               f'{LABELS[classIDs[i]]}',
                               (max(0, x), max(35, y)),
                               scale=0.5, thickness=1, colorR=(224, 182, 90),
                               colorT=(40, 40, 40),
                               font=cv2.FONT_HERSHEY_DUPLEX,
                               offset=5)

            # # draw a bounding box rectangle and label on the frame
            # color = [int(c) for c in COLORS[classIDs[i]]]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # text = "{}: {:.4f}".format(LABELS[classIDs[i]],  ################
            #                            confidences[i])
            # cv2.putText(frame, text, (x, y - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            rects.append([x, y, x + w, y + h])
            classDetected_list.append(LABELS[classIDs[i]])
            confDegree_list.append(confidences[i])

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects, classDetected_list, confDegree_list)

    # # loop over the tracked objects
    # for (objectID, centroid) in objects.items():
    #     # draw both the ID of the object and the centroid of the
    #     # object on the output frame
    #     text = f"ID {objectID}"
    #     cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),  #############
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    #     cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), - 1)

    toc = time.time()
    curr_fps = 1.0 / (toc - tic)

    # calculate an exponentially decaying average of fps number
    fps_ = curr_fps if fps_ == 0.0 else (fps_ * 0.95 + curr_fps * 0.05)
    tic = toc
    nframe = show_fps(frame, fps_)

    # draw roi
    output_frame = draw_roi(nframe, points_polygon)
    resized = cv2.resize(output_frame, (1200, int(output_frame.shape[0] * 1200 / output_frame.shape[1])))

    # # save the video with the tracking objects
    # if writer is None:
    #     fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #     writer = cv2.VideoWriter('yolov3.avi', fourcc, 20, (frame.shape[1], frame.shape[0]), True)
    # writer.write(output_frame)

    # show the output frame
    cv2.imshow('Frame', resized)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()
