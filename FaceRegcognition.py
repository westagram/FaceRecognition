import numpy as np
import face_recognition as fr
from cv2 import cv2

videoCapture = cv2.VideoCapture(0)

wesImage = fr.load_image_file("wes.jpg")
wesFaceEncoding = fr.face_encodings(wesImage)[0]
crystalImage = fr.load_image_file("crystal.jpg")
crystalFaceEncoding = fr.face_encodings(crystalImage)[0]

knownFaceEncodings = [wesFaceEncoding, crystalFaceEncoding]
knownFaceNames = ["Wes", "Toilet"]

while True:
    ret, frame = videoCapture.read()
    
    rgbFrame = frame[:, :, ::-1]
    
    faceLocations = fr.face_locations(rgbFrame)
    faceEncodings = fr.face_encodings(rgbFrame, faceLocations)

    for (top, right, bot, left), faceEncoding in zip(faceLocations, faceEncodings):
        
        matches = fr.compare_faces(knownFaceEncodings, faceEncoding)

        name = "Unknown"

        faceDistances = fr.face_distance(knownFaceEncodings, faceEncoding)

        bestMatchIndex = np.argmin(faceDistances)
        if matches[bestMatchIndex]:
            name = knownFaceNames[bestMatchIndex]

        # Face Rectangle
        cv2.rectangle(frame, (left, top), (right, bot), (0,0,255), 2)
        # Name Rectangle
        cv2.rectangle(frame, (left, bot-35), (right, bot), (0,0,255), cv2.FILLED)
        # Face Text
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, name, (left+6, bot-6), font, 1.0, (255,255,255), 1)

    cv2.imshow("Webcam FaceRecognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

videoCapture.release()
cv2.destroyAllWindows()



