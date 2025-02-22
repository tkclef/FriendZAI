import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from numpy.core.defchararray import center

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)


while True:
    success, img = cap.read()

#isSkeletalTracer -> 1
#FlippedHand_FUNCTION -> 1
    hands,img = detector.findHands(img)

#!FlippedHand_FUNCTION -> 0
#hands, img = detector.findHands(img, flipType = False)

#!SkeletalTracer  -> 0
#hands = detector.findHands(img, draw = False)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"] #List of 21 Hand Landmarks points
        bbox1 = hand1["bbox"] #Bounding BoX info x,y,w,h
        centerPoint1 = hand1["center"] #Center of the hand cx, cy
        handType1 = hand1["type"] #Hand type Left || Right

        #print(len(lmList1), lmList1)
        #print(bbox1)
        #print(centerPoint1)
        #print(handType1)

        fingers1 = detector.fingersUp(hand1)
        #length, info, img = detector.findDistance(lmList1[8], lmList1[12], img) #DistanceTrace -> 1
        #length, info = detector.findDistance(lmList1[8], lmList1[12]) #!DistanceTrace -> 0

        if len(hands)==2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Hand Landmarks points
            bbox2 = hand2["bbox"]  # Bounding BoX info x,y,w,h
            centerPoint2 = hand2["center"]  # Center of the hand cx, cy
            handType2 = hand2["type"]  # Hand type Left || Right

            fingers2 = detector.fingersUp(hand2)
            #length, info, img = detector.findDistance(lmList1[8], lmList2[8], img) #DistanceTrace -> 1
            #length, info, img = detector.findDistance(centerPoint1[8], centerPoint2[8], img)  # DistanceTrace -> 1

            #print(fingers1, fingers2) || DEBUGGING PURPOSES

    cv2.imshow("image", img)
    cv2.waitKey(1)
