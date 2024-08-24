import cv2 as cv

# --------- Mask Import For Croping Image ------------
parking_lot = cv.imread("./data/car_parking/image.png")
cv.resize(parking_lot, (640, 480))
mask = cv.imread("./data/car_parking/mask.png")

# make image black and white
mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print("Number Of Parking Lots Found: ", len(contours))

# Extract Bounding BOX from connected components
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(parking_lot, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow("Parking Lot Masking", parking_lot)
cv.waitKey(0)
