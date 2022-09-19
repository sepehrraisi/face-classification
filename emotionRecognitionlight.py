# Importing required packages
from keras.models import load_model
import numpy as np
import argparse
import dlib
import cv2
from playsound import playsound
import time
from led import smile, normal, sad

emotion_offsets = (20, 40)
emotions = {
    0: {
        "emotion": "Angry",
        "color": (193, 69, 42)
    },
    1: {
        "emotion": "Disgust",
        "color": (164, 175, 49)
    },
    2: {
        "emotion": "Fear",
        "color": (40, 52, 155)
    },
    3: {
        "emotion": "Happy",
        "color": (23, 164, 28)
    },
    4: {
        "emotion": "Sad",
        "color": (164, 93, 23)
    },
    5: {
        "emotion": "Suprise",
        "color": (218, 229, 97)
    },
    6: {
        "emotion": "Neutral",
        "color": (108, 72, 200)
    }
}


def shapePoints(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rectPoints(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


faceLandmarks = "faceDetection/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

emotionModelPath = 'models/emotionModel.hdf5'  # fer2013_mini_XCEPTION.110-0.65
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1);
cap.set(cv2.CAP_PROP_FPS, 1);
cap.set(cv2.CAP_PROP_POS_FRAMES , 1);
counter = 0
count = 0
frame_rate = 1
prev = 0
happy = False





while True:
    emotion_label_arg = 0
    emotion_prediction = 0

    # count += 1
    # if count == 2:
    #     time.sleep(1)
    #     count = 0
    time_elapsed = time.time() - prev
    ret, frame = cap.read()
    if happy:
        happy = False
        continue
    if time_elapsed > 1. / frame_rate:
        prev = time.time()

        if not ret:
            break

        scale_percent = 50  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        # frame = cv2.resize(frame, (720, 480))
        frame = cv2.resize(frame, dim)
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(grayFrame, 0)
        for rect in rects:
            shape = predictor(grayFrame, rect)
            points = shapePoints(shape)
            (x, y, w, h) = rectPoints(rect)
            grayFace = grayFrame[y:y + h, x:x + w]
            try:
                grayFace = cv2.resize(grayFace, (emotionTargetSize))
            except:
                continue

            grayFace = grayFace.astype('float32')
            grayFace = grayFace / 255.0
            grayFace = (grayFace - 0.5) * 2.0
            grayFace = np.expand_dims(grayFace, 0)
            grayFace = np.expand_dims(grayFace, -1)
            emotion_prediction = emotionClassifier.predict(grayFace)
            emotion_probability = np.max(emotion_prediction)
            if (emotion_probability > 0.36):
                normal()
                emotion_label_arg = np.argmax(emotion_prediction)
                print(emotions[emotion_label_arg]['emotion'])
                print(emotion_label_arg)
                if emotion_label_arg == 3:
                    counter = 0
                    smile()
                    playsound('ghashang.wav')
                    print("Khandidi")
                    happy = True
                    time.sleep(1)
            if emotion_label_arg != 3:
                counter += 1
                if counter >= 3:
                    print("bekhand")
                    counter = 0
                    sad()
                    playsound('bekhand.wav')

            # cv2.imshow("Emotion Recognition", frame)
            # k = cv2.waitKey(1) & 0xFF
            # if k == 27:
            #     break
cap.release()
cv2.destroyAllWindows()
