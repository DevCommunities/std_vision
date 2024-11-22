import cv2

# --------- Mask Import For Croping Image ------------
video = cv2.VideoCapture("./data/car_parking/video.mp4")
parking_lot = cv2.imread("./data/car_parking/image.png")
mask = cv2.imread("./data/car_parking/mask.png")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

# --------- Resize mask to video size ----------------
video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
mask = cv2.resize(mask, (video_width, video_height))

# --------- Handle Video Here ------------
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number Of Parking Lots Found: ", len(contours))

# --------- Masking Video ---------------
frame_avaialble = True
while frame_avaialble:
    frame_avaialble, frame = video.read()
    # Extract Bounding BOX from connected components
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

        cv2.imshow("Parking Lot Masking", frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()