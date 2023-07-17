import cv2

# Read class names from file
classNames = []
classFile = 'labels.txt'
with open(classFile , 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load pre-trained model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Open default camera
cap = cv2.VideoCapture(0)

while True:
    # Read frames from camera
    ret, img = cap.read()

    # Detect objects in the frame
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    # Draw bounding boxes and labels for detected objects
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId-1], (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Show output
    cv2.imshow('Output', img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
