
import cv2, time
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_glasses.xml")
smile_cascade = cv2.CascadeClassifier("lips.xml")
video = cv2.VideoCapture(0)
while  True:

    check, frame = video.read()

    print(frame)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Search the co-ordinates of the image
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05,
                                          minNeighbors=5)
    #scalefactor and minneighbour are used to results to improve our detector.
    font=cv2.FONT_HERSHEY_SIMPLEX
    for x, y, w, h in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (x + 1, y + 1), font, 1, (0,255, 0), 2, cv2.LINE_AA)
        #eye haarcascade
        roi_gray = gray_img[y:y+h , x:x+w]
        roi_img =frame[y:y+h , x:x+w]
        eye = eye_cascade.detectMultiScale(roi_gray)
        smile = smile_cascade.detectMultiScale(roi_gray ,scaleFactor=5)
        for ex, ey, ew, eh in eye:

            cv2.rectangle(roi_img, (ex,ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            cv2.putText(roi_img, 'eye', (ex + 1, ey + 1), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        #smile haarcascade

        for sx, sy, sw, sh in smile:
            cv2.rectangle(roi_img, (sx, sy), (sx + sw, sy + sh), (255, 0, ), 2)
            cv2.putText(roi_img, 'lips', (sx + 1, sy + 1), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    cv2.imshow('Sanjana'  , frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()