# Importing required packages
from keras.models import load_model
import numpy as np
import argparse
import dlib
import cv2
import face_recognition
import glob
import pickle
from playsound import playsound
import time
from led import smile, normal, sad


f=open("ref_name.pkl","rb")
ref_dictt=pickle.load(f)
f.close()

f=open("ref_embed.pkl","rb")
embed_dictt=pickle.load(f)
f.close()
known_face_encodings = []
known_face_names = []
for ref_id , embed_list in embed_dictt.items():
    for my_embed in embed_list:
        known_face_encodings +=[my_embed]
        known_face_names += [ref_id]
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

playsound('Names/sepehr.mp3')

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

counter = 0
count = 0
countt = 0
frame_rate = 1
prev = 0
happy = False

whois = ""
while True:
    normal()
    ret, frame = cap.read()

    time_elapsed = time.time() - prev
    ret, frame = cap.read()
    if happy:
        happy = False
        continue
    if time_elapsed > 5:
        check = True
        prev = time.time()

    if not ret:
        break

    # if not process_this_frame:
    #     break


    frame = cv2.resize(frame, (720, 480))
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
            emotion_label_arg = np.argmax(emotion_prediction)
            print(emotions[emotion_label_arg]['emotion'])
            # countt =+ 1

            if check == True:
            # if countt == 3:
            # if True:
                rgb_small_frame = frame[:, :, ::-1]
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:

                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    if name == "Unknown":
                        whois = ''
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        whois = ref_dictt[name]
                        print(whois)
                    face_names.append(name)
                    check = False
                    # countt = 0
            if emotions[emotion_label_arg]['emotion'] == "Happy":
                smile()
                if whois != "":
                    playsound(f"Names/{whois}.mp3")
                    whois = ""
                    playsound('Emotions/Smile.mp3')
                happy = True
                # time.sleep(1)
        happy = True

    cv2.imshow("Emotion Recognition", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
