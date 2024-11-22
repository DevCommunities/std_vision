import cv2
import numpy as np

# Face Detection Using Haar Cascades Algorithm
haar_cascade = cv2.CascadeClassifier('./model/haar_face.xml')
capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)
while True:
    ret, image = capture.read()
    cv2.imshow("Webcamp Steaming", image)
    
    # ---------- Face Detection --------------
    image_to_detect = image
    print("Detecting Faces")
    face_rectangle = haar_cascade.detectMultiScale(image_to_detect, scaleFactor=1.1, minNeighbors=10)
    print("Number Of Faces Found", len(face_rectangle))
    for (x,y,w,h) in face_rectangle: # Draw Reactangle Around Detected Face
        cv2.rectangle(image_to_detect, (x,y), (x+w, y+h), (0,255,0), thickness=2)
        # Add Text To The Detected Face
        cv2.putText(image_to_detect, "Detected Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    cv2.imshow("Detected Faces", image_to_detect)
    # ---------- Closing Logic ----------------
    if cv2.waitKey(1) == ord('q'):
        break
        
capture.release()
cv2.destroyAllWindows()
