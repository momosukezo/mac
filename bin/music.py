import io
import tempfile
import requests
import cv2
import numpy as np
import spotipy 
import os 
import spotipy.util as util 
from spotipy.oauth2 import SpotifyOAuth 
from spotipy.oauth2 import SpotifyClientCredentials 
from matplotlib import pyplot as plt

jacketimg = cv2.imread('music.jpeg')
flag = 0
id_now = 999


# 関数（後に別ファイル予定）
def imread_web(url):
    # 画像をリクエストする
    res = requests.get(url)
    img = None
    # Tempfileを作成して即読み込む
    with tempfile.NamedTemporaryFile(dir='./') as fp:
        fp.write(res.content)
        fp.file.seek(0)
        img = cv2.imread(fp.name)
    return img


auth_manager = SpotifyClientCredentials()
UserName = '92f9xv9b8m0zc82akd321l5dc'

MemberList = []
with open('../resister/resister_face.txt') as f:
  s = list(f)
for i in range(len(s)):
  MemberList.append(s[i].replace('\n',''))
#MemberList = list(set(MemberList))
print('メンバーリストは'+str(MemberList))

auth_manager = SpotifyClientCredentials() 
flag = 0
#img2 = cv2.imread("nana_komatsu.jpeg")


token = util.prompt_for_user_token( 
    username=UserName,  #ユーザーネームを送信 
    scope = 'app-remote-control streaming user-read-playback-state user-modify-playback-state user-read-currently-playing user-library-read user-read-playback-state playlist-read-private user-read-recently-played playlist-read-collaborative playlist-modify-public playlist-modify-private',#socope(どんな情報を得るか)の設定 
    client_id=auth_manager.client_id, #Client IDの送信 
    client_secret=auth_manager.client_secret, #Secret Client IDの送信 
    redirect_uri ='http://localhost:8889/callback' #リダイレクト先のURLを指定 
) 

sp = spotipy.Spotify(auth=token) #tokenの認証 
devices = sp.devices() 
#print(devices)
device_ids = devices["devices"][0]['id'] 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('../trainer/trainer.yml')
cascadePath = "../cascade/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


# song survey
arthistList = []
with open('../resister/resister_arthist.txt') as f:
  s = list(f)
#print(s)
for i in range(len(s)):
  arthistList.append(s[i].replace('\n',''))
#arthistList = list(set(arthistList))
print('アーティストリストは'+str(arthistList))

#results = sp.search(q=search_str, limit=1,type='playlist')
#result = sp.search(q=search_str, limit=3)
#
#for i, list in enumerate(results['playlists']['items']):
#   print('URL is ' + list['external_urls']['spotify'])
#   #print(list)

font = cv2.FONT_HERSHEY_SIMPLEX


#iniciate id counter
id = 0
print(arthistList[0])

# names related to ids: example ==> Marcelo: id=1,  etc
names = MemberList

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 2560) # set video widht
cam.set(4, 1920) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

#print('idの長さは'+len(id))

while True:

    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vertically

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
        
        #if flag == 0:
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
        # 登録された顔を検出できた場合
            print(id)
            print(names)
            print('if文成功したよ！！！！')
            try: 
              name = names[id]
            except IndexError:
              print('IndexError_face¥n¥n¥n¥n')
              id = id_now
            #idid = names.index(name)
              
            confidence = "  {0}%".format(round(100 - confidence))
            # 描画フェーズ
              
              
              
            if id != id_now:
            # 楽曲再生フェーズ
              #try:
              search_str = arthistList[id]
              #except IndexError:
              #print('IndexError_Arthist¥n¥n')
              #id = id_now
              result = sp.search(q=search_str, limit=1,type='playlist')
              for i, song in enumerate(result['playlists']['items']):
                url = song['external_urls']['spotify']
                print(url)
                sp.start_playback(device_id = device_ids, context_uri = url)
                print(str(song['images'][0]['url']))
                  
                jacketimg = imread_web(str(song['images'][0]['url']))
                # print('２枚めのタイプは'+str(type(jacketimg)))
                height, width = jacketimg.shape[:2]
                jacketimg2 = cv2.resize(jacketimg, (int(width*0.5), int(height*0.5)))
                height2, width2 = jacketimg2.shape[:2]
                img[0:height2, 0:width2] = jacketimg2
                  
                #cv2.imshow('image',jacketimg2)
                id_now = id
                
        else:
          print('if文失敗してるよ！！！!!!!!!!!!!!!!!!!!!!!!!!!!!！！')
        # 登録された顔を検出できなかった場合         
          name = "unknown"
          confidence = "  {0}%".format(round(100 - confidence))
        
          print(name)

        cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1) 
    height, width = jacketimg.shape[:2]
    jacketimg2 = cv2.resize(jacketimg, (int(width*0.5), int(height*0.5)))
    height2, width2 = jacketimg2.shape[:2]
    img[0:height2, 0:width2] = jacketimg2
    
    cv2.imshow('camera',img)

    k = cv2.waitKey(1) & 0xff # Press 'q' for exiting video
    prop_val = cv2.getWindowProperty("frame",cv2.WND_PROP_ASPECT_RATIO)
    if k == ord("q"):# or (prop_val < 0):
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")



cam.release()
cv2.destroyAllWindows()
