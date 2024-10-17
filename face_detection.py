import cv2
import numpy as np
from cvu.detector.yolov5 import Yolov5 as Yolov5Onnx

class_names = ['eyesclosed', 'eyesopen', 'face']
model = Yolov5Onnx(
    classes=class_names,
    backend="onnx",
    weight="data/face .onnx",
    device="cpu",
)

# Load pre-trained model for gender and age detection
age_net = cv2.dnn.readNetFromCaffe('data/age_deploy.prototxt', 'data/age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('data/gender_deploy.prototxt', 'data/gender_net.caffemodel')

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '>60']
gender_list = ['Male', 'Female']

def predict_age_and_gender(face_image):
    blob = cv2.dnn.blobFromImage(face_image, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]
    return gender, age

def detect_and_draw(image):
    if image is None:
        print("Failed to load image.")
        return [], None

    predictions = model(image)
    predictions.draw(image)
    return predictions, image

cap = cv2.VideoCapture(0)  # Open webcam
while True:
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        print("Failed to grab frame")
        break

    predictions = detect_and_draw(frame)[0]
    if predictions:
        for prediction in predictions:
            if prediction._class_name == 'face' and prediction._confidence > 0.5:
                x1, y1, x2, y2 = prediction.bbox
                face_img = frame[int(y1):int(y2), int(x1):int(x2)]
                gender, age = predict_age_and_gender(face_img)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                overlay_text = f"{gender}, {age}"
                cv2.putText(frame, overlay_text, (int(x2)+3, int(y2) + 0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (123, 255, 0), 2)
                # frame = cv2.resize(frame, (1280, 1280))

    cv2.imshow('Webcam', frame)  # Display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
        break

cap.release()
cv2.destroyAllWindows()
