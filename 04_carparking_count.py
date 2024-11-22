import cv2

# --------- Mask Import For Croping Image ------------
parking_lot = cv2.imread("./data/car_parking/image.png")
mask = cv2.imread("./data/car_parking/mask.png")

# --------- Handle Video Here ------------
video = cv2.VideoCapture("./data/car_parking/video.mp4")
# set to the same size as the image

# --------- Masking Image ------------
# make image black and white
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
cv2.resize(mask, (parking_lot.shape[1], parking_lot.shape[0]))

parking_lot_components = cv2.connectedComponents(mask, connectivity=4, ltype=cv2.CV_32S)
print("Number Of Parking Lots Found: ", parking_lot_components[0] - 1)
# --------- Masking Video ------------
frame_avaialble = True
while frame_avaialble:
    frame_avaialble, frame = video.read()
    cv2.resize(frame, (parking_lot.shape[1], parking_lot.shape[0]))
    for component in  parking_lot_components:
        print(component)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
            
    cv2.imshow("Parking Lot Masking", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()