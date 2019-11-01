'''
detect-crop-face.py
detecta caras de imagenes y recorta esa imagen para tener las puras caras
'''

import cv2
from scipy import misc

def facechop(image):
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)
    _, width,_ = img.shape

    width_cutoff = width//2
    s1 = img[:,:width_cutoff]
    s2 = img[:, width_cutoff:]

    minisize = (s1.shape[1],s1.shape[0])
    miniframe = cv2.resize(s1, minisize)

    faces = cascade.detectMultiScale(miniframe)
    print(faces)
    dim = []
    for f in faces:
        x, y, w, h = [ v for v in f ]
        sub_face = img[y:y+h, x:x+w]

        dimensions = sub_face.shape

        dim.append(dimensions[0])
        # max([(v,i) for i,v in enumerate(my_list)])
        max_dim = max([(v,i) for i,v in enumerate(dim)])



        # if dimensions[0] > 80:
        #     face_file_name = "faces/face_" + str(y) + ".jpg"
        #     print(face_file_name,dimensions)
        #     cv2.imwrite(face_file_name, sub_face)
        #     cv2.imshow(image, img)
    for f in range(len(faces)):
        print(faces[f][2])
        print(max_dim[0])
        x, y, w, h = [ v for v in faces[f]]
        sub_face = img[y:y+h, x:x+w]
        if faces[f][2] == max_dim[0]:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
            face_file_name = "faces/face_" + str(y) + ".jpg"
            print(face_file_name,dimensions)
            cv2.imwrite(face_file_name, sub_face)
            cv2.imwrite("detected/face_"+str(y)+".jpg",img)
            # cv2.imshow(image, img)



    return
def cut_half(img):
    height, width = img.shape

    width_cutoff = width/2
    s1 = img[:,:width_cutoff]
    s2 = img[:, width_cutoff]
    # facechop(s1)
    return s1

if __name__ == '__main__':
    img = "data-set/test3.jpg"
    #Mat halfLeft = frame(Rect(0, 0, frame.cols/2, frame.rows));

    # img = cut_half(img)
    facechop(img)

    while(True):
        key = cv2.waitKey(20)
        if key in [27, ord('Q'), ord('q')]:
            break
