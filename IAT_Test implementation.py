import cv2
import numpy as np
import time

import IAT

global cap
cap = False


def SelectCam(idx):
    global selectedCam
    global cap

    if cap:
        cap.release()

    selectedCam = idx
    cap = cv2.VideoCapture(idx)


image = False
selectedCam = 0
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        frame = np.zeros((800, 600, 3), np.uint8)

    cv2.putText(frame, "Camera ID: " + str(selectedCam), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, 8)

    cv2.imshow('Select camera', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('+'):
        SelectCam(selectedCam + 1)
    elif key == ord('-'):
        SelectCam(selectedCam - 1)
    elif key == ord(' ') and ret:
        image = frame
        break


cv2.destroyWindow("Select camera")

if not ret:
    exit(1)


colours = [
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255)
]


def drawFunc(img, rect, depth=0):
    cv2.rectangle(img, (rect[1], rect[0]), (rect[1] + rect[3], rect[0] + rect[2]), colours[depth % len(colours)])


def CoverageContrast(img):
    filled = int(np.sum(img) / 255)
    total = int(img.shape[0] * img.shape[1])
    return filled / total


def RMSContrast(img):
    amt = 4
    width = img.shape[1] - 1
    height = img.shape[0] - 1
    img1D = np.array([img[0][0], img[height][0], img[0][width], img[height][width]], np.uint8)

    # amt = img.shape[0] * img.shape[1]
    # img1D = np.reshape(img, amt)

    totalLum = 0
    for i in range(amt):
        totalLum += img1D[i]
    avgLum = totalLum / amt
    totalDiff = 0
    for i in range(amt):
        lum = img1D[i]
        diff = lum - avgLum
        totalDiff += diff * diff

    return totalDiff / amt

def MichelsonContrast(img):
    mx = int(img.max())
    mn = int(img.min())

    num = (mx - mn)
    denom = int(mx + mn)

    # filled = int(np.sum(th) / 255)
    # total = int(th.shape[0] * th.shape[1])
    # perc = filled / total
    return num / denom


def threshFunc(img, returnImg=False):
    if returnImg:
        ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
        return th
    return 1 - MichelsonContrast(img)


def none(x):
    return


cv2.namedWindow("Controls")
cv2.createTrackbar("minRes", "Controls", 4, 8, none)
cv2.createTrackbar("threshold", "Controls", 50, 100, none)
cv2.createTrackbar("depth", "Controls", 4, 16, none)


prevTime = int(round(time.time() * 1000))
deltaTime = 0
delta = 0

while True:
    currTime = int(round(time.time() * 1000))
    deltaTime = currTime - prevTime
    delta = deltaTime * .001
    prevTime = currTime

    ret, image = cap.read()

    minRes = pow(2, int(round(cv2.getTrackbarPos("minRes", "Controls") - 1)))
    depth = int(round(cv2.getTrackbarPos("depth", "Controls")))
    threshold = cv2.getTrackbarPos("threshold", "Controls") * 0.01

    iat = IAT(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), minRes, threshold)
    iat.SetDrawFunc(drawFunc)
    iat.SetThreshFunc(threshFunc)

    iat.StepAll(2, True)
    iat.StepAll(4)

    mutableImg = image.copy()
    iat.DrawItems(mutableImg)

    threshImg = np.zeros(iat.GetImageSize(), np.uint8)
    iat.ToImage(threshImg)
    binImg = np.zeros((iat.GetImageSize()[0], iat.GetImageSize()[1], 3), np.uint8)
    iat.ToBinaryImage(binImg, depth, colours)
    debugImg = np.zeros(iat.GetImageSize(), np.uint8)
    iat.ToDebugImage(debugImg)

    leftImg = np.concatenate([mutableImg, cv2.cvtColor(debugImg, cv2.COLOR_GRAY2BGR)], axis=0)
    rightImg = np.concatenate([cv2.cvtColor(threshImg, cv2.COLOR_GRAY2BGR), binImg], axis=0)

    preview = np.concatenate([leftImg, rightImg], axis=1)

    if delta != 0:

        leftOffset = leftImg.shape[1] + 16

        text = "Minimum resolution: " + str(minRes) + "px"
        cv2.putText(preview, text, (leftOffset, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        text = "FPS: " + str(round(1 / delta))
        cv2.putText(preview, text, (leftOffset, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        text = "Delta: " + str(round(delta, 3)) + "s / " + str(deltaTime) + "ms"
        cv2.putText(preview, text, (leftOffset, 96), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    cv2.imshow("IAT", preview)

    key = cv2.waitKey(16)
    if key & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
