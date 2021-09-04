import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip =0 
face_data =[]
while True:
    ret, frame = cap.read()
    if ret == False:
        continue
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #Converting to Grayscale to save memory
    
    
    faces = face_cascade.detectMultiScale(frame, 1.3,5)
    faces = sorted(faces, key=lambda f:f[2]*f[3])   #Identifying largest face by ares (w*h)
    
    for face in faces[-1:]: #s according to area; -1 would be the largest face 
        x,y,w,h = face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255),2)
        offset = 10  #To have  safe region around face 
        face_section = frame[y-offset:y+h+offset, x-offset: x+w+offset]     #Extract: region of interest from the image
        face_section = cv2.resize(face_section, (100,100))

        skip  += 1
        if (skip % 10 ==0):
            face_data.append(face_section)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Face Section", face_section)
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

        
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
np.save('sneeze.npy', face_data)
print("Data Saved")

cap.release()
cv2.destroyAllWindows()