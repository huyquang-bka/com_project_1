import cv2

vid = cv2.VideoCapture('../ui/video/video.mp4')
frames = []
success = 1
count = 0
speed = 8

while success:
    success, image = vid.read()
    if success is None:
        break
    if count % speed == 0:
        frames.append(image)
    count += 1

writer = cv2.VideoWriter('../ui/video/time_lapse.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 29.98, (1280, 720))

for frame in frames:
    writer.write(frame)
writer.release()
