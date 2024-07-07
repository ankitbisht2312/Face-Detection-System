#OpenCv  - Open Source Computer Vision Library
import cv2
from random import randrange as ran


#Load Dataset
trainedDataset = cv2.CascadeClassifier('Face.xml')

#choose a image
img = cv2.imread('Single.jpg')



#Conversion to black and white(i.e grayScale)
grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#detect faces
#This detectMultiScale detects object of different sizez => no matter what the scale of object is(Small or Big)

faceCoordinates = trainedDataset.detectMultiScale(grayImg)   
#[[225 141 169 169]] => [[X,Y,Width,Height]]


for x,y,w,h in faceCoordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)



#print(faceCoordinates)
cv2.imshow('Single Person',img)
cv2.waitKey()




'''
#display the image
cv2.imshow('Single Person',grayimg)


#pause the execution of the program untill any key is pressed
cv2.waitKey()
'''

print('END OF PROGRAM')