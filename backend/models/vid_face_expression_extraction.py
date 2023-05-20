# python vid_face_expression_extraction.py

import cv2
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
    return emots



if __name__ == '__main__':
    result = extract_expression('/Users/gohyixian/Desktop/test.mov')
    for k,v in result.items():
        print(k, ":", v)






