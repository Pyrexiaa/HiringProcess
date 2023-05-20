# python vid_face_expression_extraction.py

import cv2
import numpy as np
from deepface import DeepFace


def extract_expression(path: str, frame_interval: int = 15) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
        return None

    frame_counter = 0

    count = 0
    emots = {'sad':0, 'angry':0, 'surprise':0, 'fear':0, 'happy':0, 'disgust':0, 'neutral':0}

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
                emots['sad'] = emots.get('sad', 0) + emot['sad']
                emots['angry'] = emots.get('angry', 0) + emot['angry']
                emots['surprise'] = emots.get('surprise', 0) + emot['surprise']
                emots['fear'] = emots.get('fear', 0) + emot['fear']
                emots['happy'] = emots.get('happy', 0) + emot['happy']
                emots['disgust'] = emots.get('disgust', 0) + emot['disgust']
                emots['neutral'] = emots.get('neutral', 0) + emot['neutral']
                count += 1

        frame_counter += 1
        
    # zero division
    if count == 0: count = 1
    
    for i in list(emots.keys()):
        emots[i] /= (count*100)

    dominant = 'sad'
    for i in list(emots.keys()):
        if emots[i] > emots[dominant]:
            dominant = i
    
    emots["dominant"] = dominant
    cap.release()

    # Refactor according to certain weightage
    sad_score = emots['sad']*1.3
    angry_score = emots['angry']*1.3
    surprise_score = emots['surprise']*1.4
    fear_score = emots['fear']*1.3
    happy_score = emots['happy']*1.7
    disgust_score = emots['disgust']*10
    neutral_score = emots['neutral']/1.2
    score_list = [sad_score,angry_score,surprise_score,fear_score,happy_score,disgust_score,neutral_score]
    min_value = min(score_list)
    max_value = max(score_list)
    normalized_scores = [(score - min_value) / (max_value - min_value) for score in score_list]
    mean = np.mean(normalized_scores)
    result_scores = [(-sad_score), (-angry_score), surprise_score, (-fear_score), happy_score, (-disgust_score), neutral_score]
    min_value = min(result_scores)
    max_value = max(result_scores)
    normalized_result_scores = [(score - min_value) / (max_value - min_value) for score in result_scores]
    result = np.mean(normalized_result_scores)
    print("MEAN--------",mean)
    print("RESULT-----",result)
    
    difference = abs((mean-result)/mean)*100
    # Maintain between 0-100
    if difference>50:
        difference = 50
    if mean>result:
        value = 50-difference
    else:
        value = 50+difference

    print("CONFIDENCE LEVEL-------",value)

    # return emots, value
    return value



if __name__ == '__main__':
    # result, answer= extract_expression('C:/Users/asus/Pictures/Camera Roll/video4.MOV')
    # for k,v in result.items():
    #     print(k, ":", v)
    answer= extract_expression('C:/Users/asus/Pictures/Camera Roll/video6.MOV')






