import cv2
import os

image_paths = r"D:/Lab IC/DataTdn/01.2021-left/"
# initialized a list of images
file_path = os.listdir(image_paths)[:10]
imgs = []

for i in range(len(file_path)):
    imgs.append(cv2.imread(image_paths + file_path[i]))
    imgs[i] = cv2.resize(imgs[i], (0, 0), fx=0.8, fy=0.8)
# showing the original pictures
cv2.imshow('1', imgs[0])
cv2.imshow('2', imgs[1])
cv2.imshow('3', imgs[2])
cv2.waitKey()
cv2.destroyAllWindows()

#create stitcher object
stitcher = cv2.Stitcher_create(1)
status_code, output = stitcher.stitch(imgs)
print(output)

# final output
cv2.imshow('final result', output)

cv2.waitKey(0)
