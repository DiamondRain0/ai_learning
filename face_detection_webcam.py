import cv2
from random import randrange

#****load some pre-trained data on face frontals from opencv*****
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface.xml")
#cascade classifier => detector 


#**** choose webcam to detect faces in *****
#webcam = cv2.VidepCapture("blabla")
webcam = cv2.VideoCapture(0) # select video
key = cv2.waitKey(1)

# iterate forever over frames
while True:
    #read the current frame
    succesful_frame_read, frame = webcam.read()

    # must convert to grayscale
    graycsaled_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(graycsaled_img)
    
    #draw rectangels around the faces
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,225),5)

    
    cv2.imshow("Clever Program Face Dedector",frame)
    key = cv2.waitKey(1) #auto presses key (miliseconds)

    if key==81 or key==113:
        break

webcam.release()

print("code completed")
