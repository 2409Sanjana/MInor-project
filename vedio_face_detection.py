import cv2
import time
import requests
import numpy as np
url="http://192.168.43.1:8080/shot.jpg"
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#video = cv2.VideoCapture(0)
while  True:
    img_requ=requests.get(url)
    #check , frame = video.read()
    #print(check)
    frame=cv2.imdecode(np.array(bytearray(img_requ.content),dtype=np.uint8),-1)
    print(frame)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Search the co-ordintes of the image
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05,
                                          minNeighbors=5)
    for x, y, w, h in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 250, 0), 5)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #time.sleep(3)
    cv2.imshow('Sanjana',frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()