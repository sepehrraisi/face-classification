import face_recognition
import cv2
import numpy as np
import glob
import pickle
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
cap = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:

    ret, frame = cap.read()

    small_frame = cv2.resize(frame, (720, 480))
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            print(ref_dictt[name])

    process_this_frame = not process_this_frame
    # cv2.imshow('Video', frame)

cap.release()
cv2.destroyAllWindows()