import numpy as np
import cv2
import pyttsx3 as pts


prototxt_path = 'Object_Detection\models\MobileNetSSD_deploy.prototxt'
model_path = 'Object_Detection\models\MobileNetSSD_deploy.caffemodel'

min_confidence = 0.25

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# classes=r'coco.names'
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

#image = cv2.imread(image_path)

cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()

    height, width = image.shape[0], image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (400, 400)), 0.007, (400, 400), 130)

    net.setInput(blob)
    detected_objects = net.forward()

    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0][0][i][2]

        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])

            upper_left_x = int(detected_objects[0, 0, i, 3]*width)
            upper_left_y = int(detected_objects[0, 0, i, 4]*height)
            lower_right_x = int(detected_objects[0, 0, i, 5]*width)
            lower_right_y = int(detected_objects[0, 0, i, 6]*height)

            prediction_text = f"{classes[class_index]}:{confidence:.2f}%"

            cv2.rectangle(image, (upper_left_x, upper_left_y),
                          (lower_right_x, lower_right_y), 3)

            cv2.putText(image, prediction_text, (upper_left_x, upper_left_y-15 if upper_left_y >
                        30 else upper_left_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)


    cv2.imshow("Detected Object", image)
    pts.speak(prediction_text)
    # key=cv2.waitKey(1)q
    # if key==ord('q'):
    #     break
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

