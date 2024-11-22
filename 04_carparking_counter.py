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

# ----------- Handle Video Here ----------------------
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
totalLabels = len(contours)
print("Number Of Parking Lots Found: ", totalLabels)

# ----------- Masking Video --------------------------
frame_avaialble = True

while frame_avaialble:
    frame_avaialble, frame = video.read()
    # Extract Bounding BOX from connected components
    for i in range(1, totalLabels + 1):
        x, y, w, h = cv2.boundingRect(contours[i-1])
        black_n_white_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_n_white_frame = cv2.threshold(black_n_white_frame, 90, 255, cv2.THRESH_BINARY)[1]
        # ------------ Compare Area of Parking Lot Difference ------------
        a_lot_area = black_n_white_frame[y:y+h, x:x+w] # single parking lot area
        # display parking lot area
        cv2.imshow(f"Parking Lot {i}", a_lot_area)
        # ------------ Average Value of Color ------------
        avg_color_value, _, _, _ = cv2.mean(a_lot_area)
        white_percentage = round(avg_color_value / 255, 2)
        print(f"White % of Parking Lot {i}: {white_percentage}")
        is_parked = white_percentage < 0.82 # Threshold Value
        # ------------ Draw Rectangle With Label Below ------------
        if is_parked:
            parking_lot_area = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)
            cv2.putText(parking_lot_area, f"Parking Lot {i}", (x, y + h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        else:
            parking_lot_area = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
            cv2.putText(parking_lot_area, f"Parking Lot {i}", (x, y + h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            

    cv2.imshow("Parking Lot BW", black_n_white_frame)
    cv2.imshow("Parking Lot Masking", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()