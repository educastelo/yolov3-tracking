"""
The Centroid Tracker associates objects and updates the trackable objects.
Process:
1 - Taking an initial set of object detections (such as an input set of b-box)
2 - Creating a unique ID for each of the initial detections
3 - Tracking each of the objects as they move around frames in a video,
maintaining the assignment of unique IDs
Pipeline:
1 - Accept bounding box coordinates and compute centroids (python obj 'rects')
2 - Compute Euclidean Distance between new bounding boxes and existing objetcs
3 - Update (x, y)-coordinates of existing objects
4 - Register new objetcs
5 - Deregister old or lost objects that have moved out of frame
"""

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from datetime import datetime
import sqlite3
import os


# connect to database
LOCAL_PATH = os.getcwd()
NAME_DB = '/home/eduardo/projects/dev/count_polygon_yolov3/teste.db'
PATH_DB = os.path.join(LOCAL_PATH, NAME_DB)
sqliteConnection = sqlite3.connect(PATH_DB)
cursor = sqliteConnection.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS datavisiooh (DAY, MONTH, YEAR, HOUR, TYPE, CONFIDENCE);')
sqliteConnection.commit()


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # maxDisappeared = nr of consecutive frames until deregister and object
        # maxDistance = the maximum distance between centroids
        self.nextObjectID = 1
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.classDetected = None
        self.confidence = None
        self.counterInfo = 0

    def register(self, centroid, classDetected, confidence):
        # objects = dict w/ object ID is Key and centroid coord is value
        # disappeared = nr of consecutive frames until mark object as lost
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        self.logRegister(classDetected, confidence)

    def logRegister(self, classDetected, confidence):
        year = datetime.now().strftime("%Y")
        month = datetime.now().strftime("%m")
        day = datetime.now().strftime("%d")
        time = datetime.now().strftime("%H:%M:%S")
        sqlite_insert_query = "INSERT INTO datavisiooh (DAY, MONTH, YEAR, HOUR, TYPE, CONFIDENCE) values (?, ?, ?, ?, ?, ?)"
        data_tuple = (day, month, year, time, classDetected, int(confidence*100))
        cursor.execute(sqlite_insert_query, data_tuple)
        sqliteConnection.commit()

    def closeLogRegister(self, counterInfo):
        cursor.close()

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects, classDetected, confidence):
        # this method accept a list of bounding boxes rectangles
        # if the list of input b-box is empty, increase disappeared counter
        # if maxDisappeared number is reached, deregister the object
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # array to store the input centroid coordinates of the objects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding boxes rectangles and compute centroids
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)

            # inputCentroids store the centroid of every object of frame
            inputCentroids[i] = (cX, cY)

        # if self.objects was empty, register all objetcs
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], classDetected[i], confidence[i])

        # otherwise, try to match the input centroids to existing object
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object centroids
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # D.shape[0] = registered objects
            # D.shape[1] = new objects
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            #  track the rows and column we have already examined
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                # if row or col is already used, go for the next frame
                if row in usedRows or col in usedCols:
                    continue
                # if the distance between centroids if greater than
                # maxDistance, go for the next frame
                if D[row, col] > self.maxDistance:
                    continue

                # otherwise, grab the object ID, set its new centroid and
                # reset disappeared counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # set the row and col as used
                usedRows.add(row)
                usedCols.add(col)

            # compute the row/col we have not yet examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # if we have equal or more registered objects than new objects,
            # we need to check if any of registered objects as disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # otherwise, if the number of the new objects is greater than the
            # number of existing/registered objects, register each new centroid
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], classDetected[col], confidence[col])

        # return the set of trackable objects
        return self.objects