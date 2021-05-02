import cv2
import numpy as np
from random import randrange

#send the trained data as a reference
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale = 1
fontColor = (0,0,0)
lineType = 2

while True:
    #read current frame
    frame_read, frame = webcam.read()
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img, 1.1, 20)
    if len(face_coordinates) == 0:
        cv2.putText(frame, "No face found", bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    else:
        (a,b,c,d) = face_coordinates[0]
        for (x, y, w, h) in face_coordinates:
            if w > 200:
                cv2.rectangle(frame, (x, y), (x+w, y+w), (255, 0, 0), 5)
            elif w < 100:
                cv2.rectangle(frame, (x, y), (x+w, y+w), (0, 0, 255), 5)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+w), (0, 255, 0), 5)
        cv2.putText(frame, "Face found", bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        cv2.putText(frame, str(face_coordinates[0]), (200,50), font, 0.5, fontColor, lineType)
        if c > 200:
            cv2.putText(frame, "Face is very close", (10,450), font, 1, (150,0,0), lineType)
        elif c < 100:
            cv2.putText(frame, "Face is very far", (10,450), font, 1, (0,0,150), lineType)
        else:
            cv2.putText(frame, "Face moderately close", (10,450), font, 1, (0,150,0), lineType)
    
    cv2.imshow("Clever Programmer Face Detector", frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

webcam.release()

print("Code Completed")

'''
#import an image to cv2
img = cv2.imread("myphoto.jpg")

#grayscale it since it's easier for algorithm to understand
grayscale_img = cv2.cvtColor(webcam, cv2.COLOR_BGR2GRAY)

#get the coordinates of the rectangle drawn around the face
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

#draw coordinates on the original image
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+w), (randrange(256), randrange(256), randrange(256)), 5)

#display image result
cv2.imshow("Clever Programmer Face Detector", img)
cv2.waitKey()

#print(face_coordinates)
'''
