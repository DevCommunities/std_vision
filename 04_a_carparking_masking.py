import cv2

# --------- Mask Import For Croping Image ------------
parking_lot = cv2.imread("./data/car_parking/image.png")
cv2.resize(parking_lot, (640, 480))
mask = cv2.imread("./data/car_parking/mask.png")

# make image black and white
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number Of Parking Lots Found: ", len(contours))

# Extract Bounding BOX from connected components
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(parking_lot, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv2.imshow("Parking Lot Masking", parking_lot)
cv2.waitKey(0)
