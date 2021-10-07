# ライブラリ導入
import cv2
import os

# カメラ設定
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# 変数
face_detector = cv2.CascadeClassifier('../cascade/haarcascade_frontalface_default.xml')
MemberList    = []

def has_dup(seq):
    return len(seq) != len(set(seq))
  
MemberList = []

# 顔情報登録
MemberName    = input('\n enter your username (English) end press <return> ==>  ')

for i in range(99):

  with open('../resister/resister_face.txt',mode='a') as f:
    f.write(MemberName+'\n')
  with open('../resister/resister_face.txt') as f:
    s = list(f)
  for i in range(len(s)):
    MemberList.append(s[i].replace('\n',''))
  MemberList = list(sorted(set(MemberList),key=MemberList.index))
  with open('../resister/resister_face.txt',mode='w') as f:
    for i in range(len(MemberList)):
      f.write(MemberList[i]+'\n')
  if has_dup(s) == True:
    MemberName = input('\n The username (Eglish) is is already in use. Please enter a differnt name end press <return> ==>  ')
  else:
    print('break!!!!!!!')
    break

#
#
## 顔登録情報
#MemberName    = input('\n enter your name (English) end press <return> ==>  ')
#with open('../resister/resister_face.txt',mode='a') as f:
#  f.write(MemberName+'\n')
#with open('../resister/resister_face.txt') as f:
#  s = list(f)
#for i in range(len(s)):
#  MemberList.append(s[i].replace('\n',''))
#MemberList = list(sorted(set(MemberList),key=MemberList.index))
#with open('../resister/resister_face.txt',mode='w') as f:
#  for i in range(len(MemberList)):
#    f.write(MemberList[i]+'\n')
#

# id設定（顔情報読み込みのため）
face_id = MemberList.index(MemberName)

# お気に入り歌手情報登録
ArthistList = []
favArthist = input('\n enter your favorite arthist name (English) end press <return> ==>  ')
with open('../resister/resister_arthist.txt',mode='a') as f:
  f.write(favArthist+'\n')
with open('../resister/resister_arthist.txt') as f:
  m = list(f)
for i in range(len(m)):
  ArthistList.append(m[i].replace('\n',''))
#ArthistList = list(sorted(set(ArthistList),key=ArthistList))
with open('../resister/resister_arthist.txt',mode='w') as f:
  for i in range(len(ArthistList)):
    f.write(ArthistList[i]+'\n')


print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

# カメラ起動
while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("../dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()



# 顔情報学習
import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = '../dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("../cascade/haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('../trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
