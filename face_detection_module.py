from cProfile import label
from time import time
import cv2
import mediapipe as mp
import time

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    s = time.time()
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        relative_bb = detection.location_data.relative_bounding_box
        label_id = detection.label_id
        score = detection.score
        x = int(relative_bb.xmin * image.shape[1])
        y = int(relative_bb.ymin * image.shape[0])
        w = int(relative_bb.width * image.shape[1])
        h = int(relative_bb.height * image.shape[0])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # print("Label:", label_id, "Score:", score, "Bounding box:", x, y, w, h)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    print("FPS:", 1 / (time.time() - s))
    if cv2.waitKey(1) & 0xFF == 27:
      break
cap.release()