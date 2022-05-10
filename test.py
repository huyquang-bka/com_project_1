import cv2

#read web cam and show it
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    