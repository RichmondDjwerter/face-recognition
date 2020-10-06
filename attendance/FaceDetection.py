import cv2
import numpy as np
import face_recognition

ImageRiri = face_recognition.load_image_file(r"images/Rihanna.jpg")
ImageRiri = cv2.cvtColor(ImageRiri, cv2.COLOR_BGR2RGB)
ImageTest = face_recognition.load_image_file(r"images/Rihanna test.jpg")
ImageTest = cv2.cvtColor(ImageTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(ImageRiri)[0]
encodeRiri = face_recognition.face_encodings(ImageRiri)[0]
cv2.rectangle(ImageRiri, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 225, 0), 2)

faceLocTest = face_recognition.face_locations(ImageTest)[0]
encodeTest = face_recognition.face_encodings(ImageTest)[0]
cv2.rectangle(ImageTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (0, 225, 0), 2)

results = face_recognition.compare_faces([encodeRiri], encodeTest)
faceDistance = face_recognition.face_distance([encodeRiri], encodeTest)
print(results, faceDistance)
cv2.putText(ImageRiri,f'{results}{round(faceDistance[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Rihanna", ImageRiri)
cv2.imshow("Rihanna test", ImageTest)
cv2.waitKey(0)
