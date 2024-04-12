import cv2
import os 
import face_recognition
import replicate 
from dotenv import load_dotenv
cam = cv2.VideoCapture(1)

load_dotenv()
os.environ['REPLICATE_API_TOKEN'] = os.getenv('REPLICATE_API_TOKEN')
while True:
    #captura el video frame a frame
    ret, frame = cam.read()

    #muestra el video
    cv2.imshow('Frame', frame)

    input = {
    "image": "https://replicate.delivery/pbxt/KRULC43USWlEx4ZNkXltJqvYaHpEx2uJ4IyUQPRPwYb8SzPf/view.jpg",
    "prompt": "Are you allowed to swim here?"
}
    
    

    output = replicate.run(
        "yorickvp/llava-13b:b5f6212d032508382d61ff00469ddda3e32fd8a0e75dc39d8a4191bb742157fb",
        input=input
    )
    
    print("".join(output))
#=> "Yes, you are allowed to swim in the lake near the pier. ...

    #si se presiona la tecla 'q' se cierra la ventana
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#libera la camara y cierra la ventana
cam.release()
#cierra todas las ventanas
cv2.destroyAllWindows()