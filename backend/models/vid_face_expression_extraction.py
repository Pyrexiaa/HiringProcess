# python vid_face_expression_extraction.py

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture('/Users/gohyixian/Desktop/test.mov')

if not cap.isOpened():
    print("Error opening video file")
    exit()

frame_counter = 0
frame_interval = 15  # Select every 10th frame

count = 0
emots= [0,0,0,0,0,0,0]
desc = ['sad', 'angry', 'surprise', 'fear', 'happy', 'disgust', 'neutral']

while True:
    ret, frame = cap.read()

    # no more frames available
    if not ret:
        break

    if frame_counter % frame_interval == 0:
        bgr_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame_result = DeepFace.analyze(img_path = bgr_array, actions = ["emotion"], enforce_detection=False)

        if len(frame_result) > 0:
            emot = frame_result[0]['emotion']
            emots[0] += emot['sad']
            emots[1] += emot['angry']
            emots[2] += emot['surprise']
            emots[3] += emot['fear']
            emots[4] += emot['happy']
            emots[5] += emot['disgust']
            emots[6] += emot['neutral']
            count += 1

    frame_counter += 1
    
for i in range(len(emots)):
    emots[i] /= (count*100)

maxidx=-1
for i in range(len(emots)):
    if i > emots[maxidx]:
        maxidx = i
        
cap.release()

print("Sad      :", emots[0])
print("Angry    :", emots[1])
print("Surprise :", emots[2])
print("Fear     :", emots[3])
print("Happy    :", emots[4])
print("Disgust  :", emots[5])
print("Neutral  :", emots[6])
print("Dominant :", desc[maxidx])






