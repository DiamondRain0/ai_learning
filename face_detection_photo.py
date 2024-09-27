import cv2
from random import randrange

#****load some pre-trained data on face frontals from opencv*****
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface.xml")
#cascade classifier => detector 


#**** choose an image to detect faces in *****

img = cv2.imread("superg.png") # imread => image read


#make the image grey to make code easier
graycsaled_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = trained_face_data.detectMultiScale(graycsaled_img)
# detect multi scale =>detects different sizes
#print(face_coordinates)

#**draw rectangels around the faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(226),randrange(226),randrange(226)),2)



#display the image with the faces
cv2.imshow("Clever Program Face Dedector",img)
cv2.waitKey() #wait until a key pressed


print("code completed")
