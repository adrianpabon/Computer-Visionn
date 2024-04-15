import cv2
import os 
import face_recognition
import replicate 
from dotenv import load_dotenv
import urllib.parse
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.
drive = GoogleDrive(gauth)

#uso de la camara
cam = cv2.VideoCapture(1)
ret, frame = cam.read()
cam.release()

#guardo la imagen en una rchivo 
filename = 'myImage.png'
cv2.imwrite(filename, frame)

#subo la imagen a google drive
gfile = drive.CreateFile({'title': 'myImage.png'})
gfile.SetContentFile('myImage.png')
gfile.Upload()
print('archivo subido')

#hago quel archivo sea accesible públicamente
gfile.InsertPermission({    
    'type': 'anyone',
    'value': 'anyone',
    'role': 'reader'
})

#obtengo el id del archivo
file_id = gfile['id']
download_link = f"https://drive.google.com/uc?export=download&id={file_id}"


#se carga el archivo .env
load_dotenv()
os.environ['REPLICATE_API_TOKEN'] = os.getenv('REPLICATE_API_TOKEN')

input = {
    "image": download_link,
    "top_p": 1,
    "prompt": "What do you see?",
    "max_tokens": 1024,
    "temperature": 0.2
    }

output = replicate.run(
        "yorickvp/llava-13b:b5f6212d032508382d61ff00469ddda3e32fd8a0e75dc39d8a4191bb742157fb",
        input=input
    )
    
print("".join(output))



# Reconocimiento facial
known_face_encodings = []
known_face_names = []

for filename in os.listdir('known_people'):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        image = face_recognition.load_image_file(f'known_people/{filename}')
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(filename.split('.')[0])


while True:
    #captura el video frame a frame
    cam = cv2.VideoCapture(1)
    ret, frame = cam.read()

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    #muestra el video
    cv2.imshow('Frame', frame)


    # si se presiona la tecla 'q' se cierra la ventana
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#libera la camara y cierra la ventana
cam.release()
#cierra todas las ventanas
cv2.destroyAllWindows()