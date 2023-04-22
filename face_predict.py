import cv2
import time
import argparse
import numpy  as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from konlpy.tag import Kkma
from konlpy.utils import pprint


from deep_emotion import Deep_Emotion

def text_split(sentence):
    kkma = Kkma()
    pprint(kkma.nouns(sentence))
def face_prediction(model_path, haar_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Deep_Emotion()
    net.load_state_dict(torch.load(model_path))
    net.to(device)

    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN
    angry_count = 0
    neutral_count = 0
    fear_count = 0
    happy_count = 0
    sad_count = 0
    surprise_count = 0
    rectangle_bgr = (255,255,255)
    img = np.zeros((500,500))
    text="some text"
    (text_width, text_height) = cv2.getTextSize(text,font,fontScale = font_scale, thickness=1)[0]
    text_offset_x = 10
    text_offset_y = img.shape[0]-25
    box_coords = ((text_offset_x,text_offset_y),(text_offset_x+text_width+2,text_offset_y-text_height-2))
    cv2.rectangle(img,box_coords[0],box_coords[1],rectangle_bgr,cv2.FILLED)
    cv2.putText(img, text, (text_offset_x, text_offset_y),font,fontScale = font_scale, color=(0,0,0),thickness=1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise IOError("cannot open webcam")
    timeout = time.time() + 10  # 10 s
    while time.time() < timeout:

        negative_answer = 'negative'
        positive_answer = 'positive'

        ret, frame = cap.read()
        faceCascade = cv2.CascadeClassifier(haar_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,1.1,4)
        face_roi = None 
        for x,y,w,h in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            facess = faceCascade.detectMultiScale(roi_gray)
            if len(facess) == 0:
                print("Face not detected")
            else: 
                for (ex,ey,ew,eh) in facess:
                    face_roi = roi_color[ey: ey+eh, ex:ex + ew]
        if face_roi is not None:
            graytemp = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            final_image = cv2.resize(graytemp,(48,48))
            final_image = np.expand_dims(final_image,axis=0)
            final_image = np.expand_dims(final_image,axis=0)
            final_image = final_image/255.0
            dataa = torch.from_numpy(final_image)
            dataa = dataa.type(torch.FloatTensor)
            dataa = dataa.to(device)
            outputs = net(dataa)
            Pred = F.softmax(outputs, dim=1)
            Prediction = torch.argmax(Pred)
            print(Prediction)

            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = 1.5
            font = cv2.FONT_HERSHEY_PLAIN

            if((Prediction)==0):
                status = 'anger'
                x1,y1,w1,h1 = 0,0,175,75
                cv2.rectangle(frame,(x1, x1),(x1+w1, y1+h1), (0,0,0),-1)
                cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                cv2.putText(frame, status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
                angry_count = angry_count+1

            elif((Prediction)==1):
                status = 'neutral'
                x1,y1,w1,h1 = 0,0,175,75
                cv2.rectangle(frame,(x1, x1),(x1+w1, y1+h1), (0,0,0),-1)
                cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                cv2.putText(frame, status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
                neutral_count = neutral_count+1
        
            elif((Prediction)==2):
                status = 'fear'
                x1,y1,w1,h1 = 0,0,175,75
                cv2.rectangle(frame,(x1, x1),(x1+w1, y1+h1), (0,0,0),-1)
                cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                cv2.putText(frame, status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
                fear_count = fear_count+1

            elif((Prediction)==3):
                status = 'happy'
                x1,y1,w1,h1 = 0,0,175,75
                cv2.rectangle(frame,(x1, x1),(x1+w1, y1+h1), (0,0,0),-1)
                cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                cv2.putText(frame, status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0))
                happy_count = happy_count+1
                
            elif((Prediction)==4):
                status = 'sad'
                x1,y1,w1,h1 = 0,0,175,75
                cv2.rectangle(frame,(x1, x1),(x1+w1, y1+h1), (0,0,0),-1)
                cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                cv2.putText(frame, status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
                sad_count = sad_count+1

            elif((Prediction)==5):
                status = 'surprised'
                x1,y1,w1,h1 = 0,0,175,75
                cv2.rectangle(frame,(x1, x1),(x1+w1, y1+h1), (0,0,0),-1)
                cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                cv2.putText(frame, status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0))
                surprise_count = surprise_count+1
        else:
            print('cv2.cvtColor function does not exist')
        # cv2.imshow('Face Emption Recognition', frame)
        # if cv2.waitKey(2) & 0xFF==ord('q'):
        #     break
    cap.release()
    cv2.destroyAllWindows()
    if (angry_count+fear_count+sad_count+neutral_count)>(surprise_count+happy_count):
        return negative_answer , angry_count+fear_count , surprise_count+ happy_count, sad_count + neutral_count
    else:
        return positive_answer, angry_count+fear_count , surprise_count+ happy_count, sad_count + neutral_count