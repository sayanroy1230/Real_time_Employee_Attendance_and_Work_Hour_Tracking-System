import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import dlib
import os
import pywt   

detector = dlib.get_frontal_face_detector()

def save_cropped_image(imgpath):
    img=cv.imread(imgpath)
    gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = detector(gray_img)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        roi_color = img[y:y+h, x:x+w]
        return roi_color
path_to_model="model"
path_to_cropped_model="cropmodel"
for celeb_name in os.listdir(path_to_model):
    if (not os.path.exists(os.path.join(path_to_cropped_model,celeb_name))):
        os.mkdir(path_to_cropped_model+"/"+celeb_name)
        i=0
        for image in os.listdir(os.path.join(path_to_model,celeb_name)):
            org_img_path=os.path.join(path_to_model,celeb_name,image)
            cropimg=save_cropped_image(org_img_path)
            if cropimg is not None:
                i+=1
                img_name=path_to_cropped_model+"\\"+celeb_name+"\\"+celeb_name+"_"+str(i)+".jpg"
                cv.imwrite(img_name,cropimg)

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv.cvtColor(imArray,cv.COLOR_BGR2GRAY)
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)

    return imArray_H
celeb_dict={}
count=0
for cropmodel in os.listdir(path_to_cropped_model):
    celeb_dict[cropmodel]=count
    count+=1
# print(celeb_dict)

x=[]
y=[]
for cropmodel in os.listdir(path_to_cropped_model):
    for img in os.listdir(os.path.join(path_to_cropped_model,cropmodel)):
        raw_img=cv.imread(os.path.join(path_to_cropped_model,cropmodel,img))
        scaled_raw_img=cv.resize(raw_img,(32,32))
        wt_img=w2d(raw_img,'db1',5)
        scaled_wt_img=cv.resize(wt_img,(32,32))
        # print(col_img.shape,wt_img.shape)
        full_img=np.vstack((scaled_raw_img.reshape(32*32*3,1),scaled_wt_img.reshape(32*32,1)))
        x.append(full_img)
        y.append(celeb_dict[cropmodel])
x=np.array(x).reshape(len(x),4096).astype(float)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

x_train,x_test,y_train,y_test=tts(x,y,random_state=0)

model_params = {
    'svm': {
        'model': SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'logisticregression__C': [1,5,10]
        }
    }
}

scores = []
best_estimators = {}
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf =  GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(x_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)
print(best_estimators)
best_clf=best_estimators['svm']

from joblib import dump,load
dump(best_clf,'classifier_model.joblib')

import json
with open('celeb_dict.json',"w") as f:
    f.write(json.dumps(celeb_dict))