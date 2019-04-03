# -*- coding: utf-8 -*-
"""Scripts Version VideoLysis.

This module demonstrates the scripts version of VideoLysis
can used for developement mode.

Example:
    You can run this module this way::

        $ python scrips.py
        $ writedata()
        $ traindata()
        $ recondata()



Error case : cheeck if webcam is free to use.

Attributes:
    id (int): id of user data in.

    name (str): Name of user data in.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import numpy as np
import cv2
import os
import os.path
from PIL import Image
import json







face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.createLBPHFaceRecognizer()
cap = cv2.VideoCapture(0)
datapath = 'sdata.json'
statisticspath = 'user.json'

def doesFileExists(filePathAndName):
    """

    :param filePathAndName:
    :return:
    """
    return os.path.exists(filePathAndName)

if doesFileExists(datapath):
    json_data=open(datapath)
    statistics = json.load(json_data)
else:
    statistics = {"default": 1}
    with open(datapath, 'w') as outfile:
        json.dump(statistics, outfile)

if doesFileExists(statisticspath):
    json_data=open(statisticspath)
    docs = json.load(json_data)
else:
    docs = { 1 : 'rawand ' }
    with open(statisticspath, 'w') as outfile:
        json.dump(docs, outfile)





def write_data():
    """

    :return:
    """
    id = len(os.listdir('./data')) + 1
    print(id)
    nb = 0 
    while 1:
        ret, img = cap.read()
        only_face = np.array(10)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            nb= nb + 1
            if nb ==1:
                os.system(r'mkdir ./data/user{0}'.format(id))
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            only_face = gray[y:y+h,x:x+w]
            cv2.imwrite("./data/user"+str(id)+"/"+str(nb)+".jpg", only_face)
        cv2.imshow('live video',img)
        cv2.waitKey(1)
        if nb == 20:
            
            cv2.destroyAllWindows()
            break
            
    while 1 :
        choice = input('you know is person : yes/no')
        if choice == 'yes' or choice == 'no':
            break
    if choice == 'yes':
        ch = input('name this person')
    else:
        ch = 'inconnu'
    with open(statisticspath, "r") as jsonFile:
        docs = json.load(jsonFile)
        
    docs[id] = ch

    with open(statisticspath, "w") as jsonFile:
        json.dump(docs, jsonFile)


def train_data():
    images = []
    labels =[]
    dirs = os.listdir('data')
    for dir in dirs:
        nbimage = len(os.listdir('data/{0}'.format(dir)))
        for i in range(nbimage):
            face = cv2.imread('data/{0}/{1}.jpg'.format(dir,i+1))
            image_pil = Image.open('data/{0}/{1}.jpg'.format(dir,i+1)).convert('L')
            image = np.array(image_pil, 'uint8')
            nbr = int(dir[4:])
            faces = face_cascade.detectMultiScale(image)
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(nbr)
                cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
                cv2.waitKey(10)
    face_recognizer.train(images, np.array(labels))
    face_recognizer.save('trainer/trainer.yml')
    cv2.destroyAllWindows()
            

def recon_data():
    while True:
        ret, im =cap.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
        for(x,y,w,h) in faces:
            id_user, conf = face_recognizer.predict(gray[y:y+h,x:x+w])
            cv2.rectangle(im,(x-10,y-10),(x+w+10,y+h+10),(225,255,255),2)
            name = docs[str(id_user)]
            cv2.putText(im,str(name), (x,y-15),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, 25)
            cv2.imshow('im',im)
            cv2.waitKey(10)

#write_data()
train_data()
recon_data()
