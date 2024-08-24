import cv2 as cv

# --------- Mask Import For Croping Image ------------
video = cv.VideoCapture("./data/car_parking/video.mp4")
parking_lot = cv.imread("./data/car_parking/image.png")
mask = cv.imread("./data/car_parking/mask.png")
mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

# --------- Resize mask to video size ----------------
video_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
video_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
mask = cv.resize(mask, (video_width, video_height))

# --------- Handle Video Here ------------
contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print("Number Of Parking Lots Found: ", len(contours))

# --------- Masking Video ---------------
frame_avaialble = True
while frame_avaialble:
    frame_avaialble, frame = video.read()
    # Extract Bounding BOX from connected components
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

        cv.imshow("Parking Lot Masking", frame)

    if cv.waitKey(1) == ord('q'):
        break

video.release()
cv.destroyAllWindows()