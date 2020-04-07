import cv2 , time
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade =cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
leftear_cascade = cv2.CascadeClassifier("haarcascade_mcs_leftear.xml")
nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
video = cv2.VideoCapture(0)
while True:
    font = cv2.FONT_HERSHEY_SIMPLEX
    check, frame = video.read()

    cv2.putText(frame, 'Artificial Intelligence(AI) detection test : "WORKERS SAFTEY"', (2, 50), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, 'FONTAL FACE', (2, 80), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Eyes Detection ', (2, 100), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Mouth Detection ', (2, 120), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Ears Detection', (2, 140), font, 0.6,(0,0, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, 'MEMBERS', (2, 410), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Shreya Jain', (2, 432), font, 0.5,(0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Samiksha Dubey ', (2, 450), font, 0.5,(0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Sanjana Patel', (2, 470), font, 0.5,(0, 255, 0), 1, cv2.LINE_AA)

    print(frame)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Search the co-ordinates of the image
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1,minNeighbors=10)
    ear= leftear_cascade.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=10)
    # scalefactor and minneighbour are used to results to improve our detector.

    for x, y, w, h in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, 'Frontal Face', (x , y+h+24), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        # eye haarcascade

        roi_gray = gray_img[y:y + h, x:x + w]
        roi_img = frame[y:y + h, x:x + w]
        eye = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=14)
        nose= nose_cascade.detectMultiScale(roi_gray,scaleFactor=1.1,minNeighbors=5)
        mouth= mouth_cascade.detectMultiScale(roi_gray,scaleFactor=1.5,minNeighbors=25)

        for ex, ey, ew, eh in eye:
            cv2.rectangle(roi_img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            cv2.putText(roi_img, 'eyes unprotected', (ex, ey + eh + 10), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
             # smile haarcascade

        for mx, my, mw, mh in mouth:
            cv2.rectangle(roi_img, (mx, my), (mx + mw, my + mh), (0, 0, 0), 2)
            cv2.putText(roi_img, 'Mouth Unprotected', (mx, my + mh + 10), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        for nx, ny, nw, nh in nose:
            cv2.rectangle(roi_img, (nx, ny), (nx + nw, ny + nh), (255, 255, 255), 2)
    for lex, ley, lew, leh in ear:
        img = cv2.rectangle(frame, (lex, ley), (lex + lew, ley + leh), (0,255,0), 2) #(B,G,R)
        cv2.putText(frame, 'Ear unprotected', (lex, ley + leh + 10), font, 0.5, (0,255,0), 2, cv2.LINE_AA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Sanjana', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()