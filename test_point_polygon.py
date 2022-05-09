import cv2
import numpy as np

image = cv2.imread('1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 120, 255, 1)
cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

point1 = (25, 50)
point2 = (100, 150)
point3 = (10, 100)


def check_point(polygon_array, point):
    result1 = 0
    for c in polygon_array:
        result1 = cv2.pointPolygonTest(c, point, False)

    # Draw points
    cv2.circle(image, point1, 8, (100, 100, 255), -1)
    cv2.putText(image, 'point1', (point[0] - 10, point[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0),
                lineType=cv2.LINE_AA)
    return result1


result = check_point(cnts, point1)
print(result)
