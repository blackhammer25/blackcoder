import os
import cv2
import numpy as np
from PIL import Image
import pickle


base_dir=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(base_dir,"images")

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
#recognizer=cv2.face.createLBPHFaceRecognizer()

current_id=0
label_ids={}
y_labels=[]
x_train=[]


for root,dirs,files in os.walk(image_dir):
    for i in files:
        if i.endswith("png") or i.endswith("jpg"):
            path=os.path.join(root,i)
            label=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            #print(label,path)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
            #print(label_ids)
            #my labels.append(label)some number series
            #verifying the image and turn it into a numpy list
            pil_image=Image.open(path).convert("L")#image gets converted into grayscale
            image_array=np.array(pil_image,"uint8")
            #print(image_array)#our image is converted into numbers pretty cool!!
            faces= face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            
            for (x,y,w,h) in  faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                
                
##print(y_labels)
##print(x_train)
            
with open("label.pickle",'wb') as f:
    pickle.dump(label_ids,f)
    
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")
    
    
            

    