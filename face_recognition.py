import face_recognition
import cv2
import numpy as np
import mediapipe as mp

a=int(input('if you want to detect face using media pipe press 1 or using face_recognition press 2\n\n'))

if a==1:
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("video not found")
                continue

            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = face_detection.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)
            cv2.imshow('Face ', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

elif a==2:
    video_capture = cv2.VideoCapture(0)

    chai_image = face_recognition.load_image_file("myimage.jpeg")
    chai_face_encoding = face_recognition.face_encodings(chai_image)[0]
    dinesh_image = face_recognition.load_image_file("dinesh.jpeg")
    dinesh_face_encoding = face_recognition.face_encodings(dinesh_image)[0]

    known_face_encodings = [chai_face_encoding, dinesh_face_encoding]
    known_face_names = ["chaitanya sai", "Dinesh kumar"]
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

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

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    print('you press wrong choice')

video_capture.release()
cv2.destroyAllWindows()