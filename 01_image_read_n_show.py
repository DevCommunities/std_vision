import cv2 as cv

image = cv.imread("./data/train/guide.png")

cv.imshow("Guide", image)
cv.waitKey(0)