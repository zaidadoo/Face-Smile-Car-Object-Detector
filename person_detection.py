import cv2

body_data = cv2.CascadeClassifier('haarcascade_fullbody.xml')
upperbody_data = cv2.CascadeClassifier('haarcascade_upperbody.xml')
webcam = cv2.VideoCapture(0)

while True:
    frame_read, frame = webcam.read()
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    body_coordinates = body_data.detectMultiScale(grayscale_img)
    upperbody_coordinates = upperbody_data.detectMultiScale(grayscale_img)
    for (x, y, w, h) in body_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+w), (0, 255, 0), 5)
    for (x, y, w, h) in upperbody_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+w), (0, 255, 0), 5)
    cv2.imshow("Clever Programmer Face Detector", frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

webcam.release()