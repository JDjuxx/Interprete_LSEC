#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
import mediapipe as mp
import random

def get_bool():
	if(random.random() <= 0.8):
		return True
	else:
		return False

# In[3]:


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):

    mp_drawing.draw_landmarks(image, 
                              results.right_hand_landmarks, 
                              mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, 
                              results.left_hand_landmarks, 
                              mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(234,23,10), thickness=1, circle_radius=1))
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh,rh])

def extract_keypoints_V3(results):
    rh_x = [res.x for res in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else []
    lh_x = [res.x for res in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else []
    rh_y = [res.y for res in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else []
    lh_y = [res.y for res in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else []
    if(lh_y and lh_x):
        if(min(lh_y) >= 1):
            lh_y = []
        if(min(lh_x) >= 1):
            lh_x = []
    if(rh_y and rh_x):
        if(min(rh_y) >= 1):
            rh_y = []
        if(min(rh_x) >= 1):
            rh_x = []
    return rh_x,rh_y,lh_x,lh_y

def find_max_min(keypoints_x, keypoints_y):
    x_min = int(min(keypoints_x)*480-5)
    x_max = int(max(keypoints_x)*480+5)
    y_min = int(min(keypoints_y)*360-5)
    y_max = int(max(keypoints_y)*360)
    if(x_min < 0):
        x_min = 0
    if(y_min < 0):
        y_min = 0
    if(y_max > 360):
        y_max = 360
    if(x_max > 480):
        x_max = 480
    return x_min, x_max, y_min, y_max

def hand_img(keypoints_x, keypoints_y, image):
    x_min, x_max, y_min, y_max = find_max_min(keypoints_x,keypoints_y)
    hand_image = np.array(cv2.resize(image[y_min:y_max , x_min:x_max], (60,60)))/255
    return hand_image

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


# In[ ]:



