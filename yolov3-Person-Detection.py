# import Depedencies
import cv2
import numpy as np
import time
import pandas as pd


# Video File Setup
FILE_PATH = 'Dataset/5 FPS/Video Shoot 1/'
FILE_NAME = "Abnormal 1"
FILE_EXT = ".mp4"
cap = cv2.VideoCapture(FILE_PATH + FILE_NAME + FILE_EXT)

# Load YOLOV3
YOLO_WEIGHTS_PATH = "yolov3.weights"
YOLO_CONFIG_PATH = "yolov3.cfg"
COCO_NAMES_PATH = "coco.names"
net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CONFIG_PATH)
classes = []
with open(COCO_NAMES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Set timer for FPS counter
starting_time = time.time()
frame_id = 0
font = cv2.FONT_HERSHEY_PLAIN

# Set output file (for centroid)
bounding_box = {'centroid_x' : [], 'centroid_y' : [], 'height' : []}

# Open file in Loop
while True:

    # Read file 
    ret, frame = cap.read()

    # Check file availability
    if(ret):

        # resize image
        frame_id += 1

        # get W,H and Depth (Channel) from image
        height, width, channels = frame.shape

        # Detecting objects on frame
        blob = cv2.dnn.blobFromImage(image = frame, scalefactor = 1/255.0, size = (416, 416), mean = (0,0,0), swapRB = True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        centroids = [] 

        # Loop through the detected objects
        for out in outs:
            for detection in out:

                # Get probability and class index(argmax)
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Set probability threshold and class id (0 = person)
                if confidence > 0.5 and class_id == 0:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # update bounding box properties
                    centroids.append([center_x,center_y])
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # make bounding box
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

        # Draw all bounding box in a frame
        for i in range(len(boxes)):
            if i in indexes:
                # get coordinates and centroids
                x, y, w, h = boxes[i]
                cx, cy = centroids[i]

                # Get label
                label = str(classes[class_ids[i]])

                # Draw rectangle
                confidence = confidences[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)

                # Draw circle for centroids
                cv2.circle(frame, (cx, cy) ,10, (0,255,0), -1)

                # Put label to frame
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, (0,0,255), 3)

                # Get centroid and height for output file
                bounding_box['centroid_x'].append(cx)
                bounding_box['centroid_y'].append(cy)
                bounding_box['height'].append(h)
                print(cx,cy,h)

        # Calculate FPS
        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (255, 0, 0), 3)

        # Show detected frame and the bounding boxes
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    else:
        break


# Export Centroids
OUTPUT_PATH = "Dataset/5 FPS/Video Shoot 1/bounding box/"
OUTPUT_NAME = FILE_NAME
OUT_EXT = ".xlsx"
export_centroids = pd.DataFrame(bounding_box)
export_centroids.to_excel( OUTPUT_PATH+ FILE_NAME + OUT_EXT, index = False)

# Close window
cap.release()
cv2.destroyAllWindows()