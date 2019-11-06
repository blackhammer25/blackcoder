import cv2
import pickle
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels={"person_name":1}
with open("label.pickle",'rb') as f:
    o_labels=pickle.load(f)
    labels={v:k for k,v in o_labels.items()}


video = cv2.VideoCapture('a.mp4')
while True:
    re,image= video.read()
    
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    faces=face_cascade.detectMultiScale(gray_image,1.2,4)
    
    for (x,y,w,h) in faces:
        roi_gray=gray_image[y:y+h,x:x+w]
        
        id_,conf=recognizer.predict(roi_gray)
        if conf>=100 and conf<=225:
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(0,225,0)
            stroke=1
            cv2.putText(image,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,268,0),1)
        
    cv2.imshow('ua',image)
    
    a=cv2.waitKey(30) & 0xff
    if a==ord("q"):
        break
video.release()
cv2.destroyAllWindows()        
        
        