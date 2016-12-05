'''
This demo is used to detect the eye and face from an image.
'''
import cv2
 
face_cascade = cv2.CascadeClassifier('C:\\opencv3\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\opencv3\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')
 
img = cv2.imread('C:\\data\\input_face_image.jpg')
if img is None:
    print('Failed to load the image. Exit...')
    exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imwrite('image_eye_face_detected.jpg', img)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()