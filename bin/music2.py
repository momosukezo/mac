
import cv2
import numpy as np
import os 
import requests 
import spotipy 
import os 
import spotipy.util as util 
from spotipy.oauth2 import SpotifyOAuth 
from spotipy.oauth2 import SpotifyClientCredentials 




auth_manager = SpotifyClientCredentials()
UserName = '92f9xv9b8m0zc82akd321l5dc'

MemberList = []
  with open('resister.txt') as f:
    s = list(f)
 print(s)
 for i in range(len(s)):
 MemberList.append(s[i].replace('\n',''))


auth_manager = SpotifyClientCredentials() 
flag = 0
img2 = cv2.imread("nana_komatsu.jpeg")
pl_matsu = "https://open.spotify.com/playlist/3QhsneaHkM61rVpsLJtFZm?si=d38fbe956d6d4a05"
pl_maki = "https://open.spotify.com/playlist/37i9dQZF1DWWFmkwzsTTJg?si=ecc11ef66b3d4257"
pl_okubo = "https://open.spotify.com/playlist/37i9dQZF1DZ06evO043Tqq?si=b0fffb56ac5c4689"
pl_uchino = "https://open.spotify.com/playlist/7xFPZsPMZ69wK64VvPLQ3h?si=6ce833a7c06f432e"

token = util.prompt_for_user_token( 
    username=UserName,  #ユーザーネームを送信 
    scope = 'app-remote-control streaming user-read-playback-state user-modify-playback-state user-read-currently-playing user-library-read user-read-playback-state playlist-read-private user-read-recently-played playlist-read-collaborative playlist-modify-public playlist-modify-private',#socope(どんな情報を得るか)の設定 
    client_id=auth_manager.client_id, #Client IDの送信 
    client_secret=auth_manager.client_secret, #Secret Client IDの送信 
    redirect_uri ='http://localhost:8889/callback' #リダイレクト先のURLを指定 
) 

sp = spotipy.Spotify(auth=token) #tokenの認証 
devices = sp.devices() 
print(devices)
device_ids = devices["devices"][0]['id'] 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = MemberList

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 2560) # set video widht
cam.set(4, 1920) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            if flag == 0:
               if id == "matsu":
                  sp.start_playback(device_id = device_ids, context_uri = pl_matsu)
                  flag = 10
                  img2 = cv2.imread("nana_seino.jpeg")

               elif id == "maki":
                  sp.start_playback(device_id = device_ids, context_uri = pl_maki)
                  flag = 11
               elif id == "okubo":
                  sp.start_playback(device_id = device_ids, context_url = pl_okubo)
                  flag = 12
               elif id == "uchino":
                  sp.start_playback(device_id = device_ids, context_url = pl_uchino)
                  flag = 13
                
                
                 
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1) 

#    height, width = img2.shape[:2]
#    img[0:height, 0:width] = img2
    cv2.imshow('camera',img) 

    k = cv2.waitKey(1) & 0xff # Press 'q' for exiting video
    prop_val = cv2.getWindowProperty("frame",cv2.WND_PROP_ASPECT_RATIO)
    if k == ord("q"):# or (prop_val < 0):
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")



cam.release()
cv2.destroyAllWindows()
