# python3 deepface_recognition.py

from deepface import DeepFace

image1 = 'lol1.png'
image2 = 'lol2.png'
image3 = 'lol3.png'

# first time running will download VGG-Face model (.h5) (~580mb)
# Compare if same face
print(DeepFace.verify(img1_path = image1,  # yx
                img2_path = image1,  # jy
                enforce_detection=False))
# output
    # {'verified': True,
    # 'distance': 0.17842618501190277,
    # 'threshold': 0.4,
    # 'model': 'VGG-Face',
    # 'detector_backend': 'opencv',
    # 'similarity_metric': 'cosine',
    # 'facial_areas': {'img1': {'x': 42, 'y': 61, 'w': 144, 'h': 144},
    # 'img2': {'x': 73, 'y': 57, 'w': 103, 'h': 103}},
    # 'time': 0.27
    # }

# each feature (i.e. age, gender, emotion) is a separate model, each size ~500mb
print(DeepFace.analyze(img_path = image1, 
                 actions = ["age", "gender", "emotion", "race"]))
# output
    # [{'age': 35,
    # 'region': {'x': 31, 'y': 46, 'w': 117, 'h': 117},
    # 'gender': {'Woman': 0.015357557276729494, 
    #         'Man': 99.98464584350586
    #         },
    # 'dominant_gender': 'Man',
    # 'emotion': {'angry': 0.3038950626725033,
    #             'disgust': 3.667220231060474e-11,
    #             'fear': 2.3939014472247897,
    #             'happy': 1.2440780556642484e-05,
    #             'sad': 87.49081939349405,
    #             'surprise': 6.846103949403675e-05,
    #             'neutral': 9.81130493418037
    #             },
    # 'dominant_emotion': 'sad',
    # 'race': {'asian': 7.334453304675418,
    #         'indian': 3.1661530981155095,
    #         'black': 85.50387534522267,
    #         'white': 0.09932484836949994,
    #         'middle eastern': 0.03912873741168454,
    #         'latino hispanic': 3.8570622418559934
    #         },
    # 'dominant_race': 'black'
    # }]
