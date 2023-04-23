import cv2
import dlib
import numpy as np
from keras.models import load_model
# from led import smile, normal, sad
from playsound import playsound


emotion_offsets = (20, 40)
emotions = {
    0: {"emotion": "Angry", "color": (193, 69, 42)},
    1: {"emotion": "Disgust", "color": (164, 175, 49)},
    2: {"emotion": "Fear", "color": (40, 52, 155)},
    3: {"emotion": "Happy", "color": (23, 164, 28)},
    4: {"emotion": "Sad", "color": (164, 93, 23)},
    5: {"emotion": "Surprise", "color": (218, 229, 97)},
    6: {"emotion": "Neutral", "color": (108, 72, 200)}
}

faceLandmarks = "faceDetection/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

emotionModelPath = 'models/emotionModel.hdf5'  
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 0.5)

counter = 0
happy = False

while True:
    emotion_label_arg = 0
    emotion_prediction = 0
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(frame, 0)
    for rect in rects:
        shape = predictor(frame, rect)
        points = np.zeros((5, 2), dtype="int")
        points[0] = (shape.part(17).x, shape.part(17).y)
        points[1] = (shape.part(21).x, shape.part(21).y)
        points[2] = (shape.part(22).x, shape.part(22).y)
        points[3] = (shape.part(26).x, shape.part(26).y)
        points[4] = (shape.part(16).x, shape.part(16).y)

        (x, y, w, h) = rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()
        grayFace = frame[y:y + h, x:x + w]
        try:
            grayFace = cv2.resize(grayFace, emotionTargetSize)
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
            # normal()
            emotion_label_arg = np.argmax(emotion_prediction)
            print(emotions[emotion_label_arg]['emotion'])
            print(emotion_label_arg)
            if emotion_label_arg == 3:
                counter = 0
                # smile()
                playsound('ghashang.wav')
                print("Khandidi")
                happy = True
                # time.sleep(1)
        if emotion_label_arg != 3:
            counter += 1
            if counter >= 3:
                print("bekhand")
                happy = True
                counter = 0
                # sad()
                playsound('bekhand.wav')

    if happy:
        happy = False

cap.release()
cv2.destroyAllWindows()