import cv2
import numpy as np

image = cv2.imread("./data/train/guide.png")
# cv2.imshow("Original", image)

# Face Detection Using Haar Cascades Algorithm
image_to_detect = image.copy()
haar_cascade = cv2.CascadeClassifier('./model/haar_face.xml')
face_rectangle = haar_cascade.detectMultiScale(image_to_detect, scaleFactor=1.1, minNeighbors=6)
print("Number Of Faces Found", len(face_rectangle))

for (x,y,w,h) in face_rectangle:
    cv2.rectangle(image_to_detect, (x,y), (x+w, y+h), (0,255,0), thickness=2)
    
cv2.imshow("Detected Face", image_to_detect)
cv2.waitKey(0)