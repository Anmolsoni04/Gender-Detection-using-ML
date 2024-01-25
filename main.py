import cv2 as cv #is used to capturing image
import math #used for height & width of the object
import time #to reading the face
import argparse #the arguement will pass in argparse wil be added at the time of getting output

def getFaceBox(net, frame, conf_threshold = 0.7): #frame -> opencvdnn, height, width
    frameOpencvDnn = frame.copy() #Deep Nueron network
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300,300), [104,117,123], True, False) #image, scale factor, size, mean
    
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold: 
            x1 = int(detections[0,0,i,3] * frameWidth)
            y1 = int(detections[0,0,i,4] * frameHeight)
            x2 = int(detections[0,0,i,5] * frameWidth)
            y2 = int(detections[0,0,i,6] * frameHeight)
            bboxes.append([x1,y1,x2,y2])
            cv.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0),int(round(frameHeight/150)),8)
    return frameOpencvDnn, bboxes

parser = argparse.ArgumentParser(description='USe this script to run gender recognition using openCv')

parser.add_argument('--input', help='Path to input image or video file. skip this argument to capture frame from a camera.')

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

genderList = ['Male','Female'] #List in python is mutable(i.e., it can be  changed)

#Load the deep network


genderNet = cv.dnn.readNet(genderModel,genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

#Open the live camera or image file or video file

cap = cv.VideoCapture(args.input if args.input else 0)
padding = 20

while cv.waitKey(1) < 0:
    #ReadFrame
    t = time.time()
    hasFrame,frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face detected, try again")
        continue

    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        # print("Gender Output : {}".format(genderPreds))
        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        # ageNet.setInput(blob)
        # agePreds = ageNet.forward()
        # age = ageList[agePreds[0].argmax()]
        # print("Age Output : {}".format(agePreds))
        # print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

        label = "{}".format(gender)
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow("Gender Demo", frameFace)
        # cv.imwrite("age-gender-out-{}".format(args.input),frameFace)
        
    print("time : {:.3f}".format(time.time()))  