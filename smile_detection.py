import cv2
import numpy

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_data = cv2.CascadeClassifier('haarcascade_smile.xml')
webcam = cv2.VideoCapture(0)

while True:
    frame_read, frame = webcam.read()

    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img, 1.1, 20)

    for (x, y, w, h) in face_coordinates:

        cv2.rectangle(frame, (x, y), (x+w, y+w), (0, 255, 0), 2)

        the_face = frame[y:y+h, x:x+w]

        grayscale_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smile_coordinates = smile_data.detectMultiScale(grayscale_face, 1.3, 20)

        for (x2, y2, w2, h2) in smile_coordinates:
            #cv2.rectangle(the_face, (x2, y2), (x2+w2, y2+w2), (0, 0, 255), 2)
            cv2.putText(frame, "Smiling", (int((x+(w/2))-50), y+h+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 4)
            cv2.putText(frame, "Smiling", (int((x+(w/2))-50), y+h+30), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
            
    cv2.imshow("Clever Programmer Face Detector", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()