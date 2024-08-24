import cv2 as cv

# --------- Mask Import For Croping Image ------------
parking_lot = cv.imread("./data/car_parking/image.png")
mask = cv.imread("./data/car_parking/mask.png")

# --------- Handle Video Here ------------
video = cv.VideoCapture("./data/car_parking/video.mp4")
# set to the same size as the image

# --------- Masking Image ------------
# make image black and white
mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
cv.resize(mask, (parking_lot.shape[1], parking_lot.shape[0]))

parking_lot_components = cv.connectedComponents(mask, connectivity=4, ltype=cv.CV_32S)
print("Number Of Parking Lots Found: ", parking_lot_components[0] - 1)
# --------- Masking Video ------------
frame_avaialble = True
while frame_avaialble:
    frame_avaialble, frame = video.read()
    cv.resize(frame, (parking_lot.shape[1], parking_lot.shape[0]))
    for component in  parking_lot_components:
        print(component)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
            
    cv.imshow("Parking Lot Masking", frame)
    if cv.waitKey(1) == ord('q'):
        break

video.release()
cv.destroyAllWindows()