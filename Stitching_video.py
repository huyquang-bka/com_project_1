import cv2
import os

image_paths = 'Image/'
# initialized a list of images
file_path = os.listdir(image_paths)
imgs = []

for i in range(len(file_path)):
    imgs.append(cv2.imread(image_paths+file_path[i]))
    imgs[i] = cv2.resize(imgs[i], (0, 0), fx=0.8, fy=0.8)
# showing the original pictures
cv2.imshow('1', imgs[0])
cv2.imshow('2', imgs[1])
cv2.imshow('3', imgs[2])

stitchy = cv2.Stitcher.create()
(dummy, output) = stitchy.stitch(imgs)

if dummy != cv2.STITCHER_OK:
    # checking if the stitching procedure is successful
    # .stitch() function returns a true value if stitching is
    # done successfully
    print("stitching ain't successful")
else:
    print('Your Panorama is ready!!!')

# final output
cv2.imshow('final result', output)

cv2.waitKey(0)