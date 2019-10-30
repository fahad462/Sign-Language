import cv2 as cv
import numpy as np
import util as ut
import svm_train as st
import re

model = st.trainSVM(17)

# cam = int(input("Enter Camera number: "))
cap = cv.VideoCapture(1)
#cap.set(3,1080);
#cap.set(4,720)
font = cv.FONT_HERSHEY_SIMPLEX


def nothing(x):
    pass


text = " "

temp = 0
previouslabel = None
previousText = " "
label = None
while (cap.isOpened()):
    _, img = cap.read()
    img = cv.flip(img, 1)
    # cv.rectangle(img, (900, 100), (1300, 500), (255, 0, 0),
    #              3)  # bounding box which captures ASL sign to be detected by the system

    cv.rectangle(img, (50, 100), (450, 550), (255, 0, 0), 3)

    # print(img)

    height = np.size(img, 0)
    width = np.size(img, 1)

    # print (height + ' ' + width)

    # img1 = img[100:500, 900:1300]

    img1 = img[100:550, 50:450];

    # print(img1)

    img_ycrcb = cv.cvtColor(img1, cv.COLOR_BGR2YCR_CB)
    blur = cv.GaussianBlur(img_ycrcb, (11, 11), 0)
    skin_ycrcb_min = np.array((0, 138, 67))
    skin_ycrcb_max = np.array((255, 173, 133))
    mask = cv.inRange(blur, skin_ycrcb_min,
                      skin_ycrcb_max)  # detecting the hand in the bounding box using skin detection
    contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, 2)
    cnt = ut.getMaxContour(contours, 2000)  # using contours to capture the skin filtered image of the hand\
    # print(cnt)
    if cnt is not None:
        gesture, label = ut.getGestureImg(cnt, img1, mask,
                                          model)  # passing the trained model for prediction and fetching the result
        if (label is not None):
            if (temp == 0):
                previouslabel = label
        if previouslabel == label:
            previouslabel = label
            temp += 1
        else:
            temp = 0
        if (temp == 40):
            if (label == 'P'):
                label = " "
            text = text + label
            if (label == 'Q'):
                words = re.split(" +", text)
                words.pop()
                text = " ".join(words)
                # text=previousText
            # print(str(label))

        # cv.imshow('PredictedGesture', gesture)  # showing the best match or prediction
        cv.putText(img, label, (50, 150), font, 8, (0, 125, 155),
                   2)  # displaying the predicted letter on the main screen
        cv.putText(img, text, (50, 450), font, 3, (0, 0, 255), 2)
        # print('upto ' + label)
    cv.imshow('Frame', img)
    cv.imshow('Mask', mask)
    k = 0xFF & cv.waitKey(10)
    if k == ord('a'):
        break

cap.release()
cv.destroyAllWindows()
