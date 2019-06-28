#imports
import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

#file-import: haar-basis cascades
cascPath = "haarcascade_frontalface_default.xml"

#open CV's in-built prediction module
faceCascade = cv2.CascadeClassifier(cascPath)

#logging the event-information in a seoarate file
log.basicConfig(filename='webcam.log',level=log.INFO)

# This line sets the video source to the default webcam
# You can also provide a filename here, and Python will read in the video file
    # However, you need to have ffmpeg installed for that since OpenCV itself cannot decode compressed video
    # ffmpeg acts as the front end for OpenCV, and, ideally, it should be compiled directly into OpenCV
video_capture = cv2.VideoCapture(0)

anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    # The read() function reads one frame from the video source, which in this example is the webcam. This returns:
        # The actual video frame read (one frame on each loop)
        # A return code
            # The return code tells us if we have run out of frames, which will happen if we are reading from a file
                # This doesn’t matter when reading from the webcam, since we can record forever
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detectMultiScale: Detects objects of different sizes in the input image
    #the detected objects are returned as a list of rectangles
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,    #Parameter specifying how much the image size is reduced at each image scale
        minNeighbors=5,     #Parameter specifying how many neighbors each candidate rectangle should have to retain it
        minSize=(30, 30)    #Minimum possible object size: Objects smaller than that are ignored
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)

    # We wait for the ‘q’ key to be pressed. If it is pressed at any time, then we exit the script
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
