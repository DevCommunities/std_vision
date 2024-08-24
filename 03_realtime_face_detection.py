import cv2 as cv
import numpy as np

# Face Detection Using Haar Cascades Algorithm
haar_cascade = cv.CascadeClassifier('./model/haar_face.xml')
capture = cv.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)
while True:
    ret, image = capture.read()
    cv.imshow("Webcamp Steaming", image)
    
    # ---------- Face Detection --------------
    image_to_detect = image
    print("Detecting Faces")
    face_rectangle = haar_cascade.detectMultiScale(image_to_detect, scaleFactor=1.1, minNeighbors=10)
    print("Number Of Faces Found", len(face_rectangle))
    for (x,y,w,h) in face_rectangle: # Draw Reactangle Around Detected Face
        cv.rectangle(image_to_detect, (x,y), (x+w, y+h), (0,255,0), thickness=2)
        # Add Text To The Detected Face
        cv.putText(image_to_detect, "Detected Face", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    cv.imshow("Detected Faces", image_to_detect)
    # ---------- Closing Logic ----------------
    if cv.waitKey(1) == ord('q'):
        break
        
capture.release()
cv.destroyAllWindows()
