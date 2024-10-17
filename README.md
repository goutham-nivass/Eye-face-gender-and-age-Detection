
# Real-Time Face Detection and Age-Gender Prediction

## Overview
This project utilizes OpenCV and YOLOv5 (via CVU) to perform real-time face detection and age-gender prediction on video streams. The system reads frames from a video file, detects faces, predicts the age and gender of each detected face, and overlays the results on the video for real-time viewing.

## Key Features
- **Face Detection**: Leverages a YOLOv5 ONNX model to detect faces in video frames.
- **Age and Gender Prediction**: Uses pre-trained models from Caffe to estimate the age and gender of detected faces.
- **Real-Time Video Processing**: Displays the processed video stream in real-time with age and gender annotations.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- CVU Library for YOLOv5 integration

## Setup and Installation
1. **Clone the repository**:
   ```
   git clone https://github.com/a-pragatheeswaran/eye-face-gender-and-age-Detection.git
   ```
2. **Install dependencies**:
   ```
   pip install opencv-python numpy cvu
   ```

## Usage
To run the project, execute the main script via the command line:
```
python face_detection.py
```

This script opens a video stream, applies face detection and age-gender prediction, and displays the annotated video in real-time. Press 'q' to quit the video stream window.

## Code Explanation

### Video Capture
The video is captured from a video file:
```python
cap = cv2.VideoCapture("/path/to/video.mp4")
```

### Frame Processing
Each frame is processed to detect faces and predict age and gender:
```python
while True:
    ret, frame = cap.read()
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
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2), (0, 255, 0), 2)
                overlay_text = f"{gender}, {age}"
                cv2.putText(frame, overlay_text, (int(x2)+3, int(y2) + 0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (123, 255, 0), 2)

    cv2.imshow('Webcam', frame)  # Display the processed frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```


