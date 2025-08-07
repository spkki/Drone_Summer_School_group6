import cv2
import numpy as np
import random
from icecream import ic

def findContoursInFile(filename):
    image = cv2.imread(filename, 1)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(image_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def analyzeContours(contours, objectClass):
    observations = []
    for idx, contour in enumerate(contours):
        observation = []
        perimeter = cv2.arcLength(contours[idx], 1)
        area = cv2.contourArea(contours[idx], True)
        if area <= 0:
            continue
        compactness = perimeter * perimeter / (4 * 3.141592 * area)

        mu = cv2.moments(contours[idx], False)
        hu = cv2.HuMoments(mu).flatten()
        coordinatex = mu['m10'] / mu['m00']
        coordinatey = mu['m01'] / mu['m00']
        observation.append(objectClass)
        observation.append(coordinatex)
        observation.append(coordinatey)
        observation.append(area)
        observation.append(perimeter)
        observation.append(mu['mu12'] / 1000.)
        observations.append(observation)
    return observations


def main():
    t1 = analyzeContours(findContoursInFile("../input/numbers/1.png"), 1)
    t2 = analyzeContours(findContoursInFile("../input/numbers/2.png"), 2)
    t3 = analyzeContours(findContoursInFile("../input/numbers/3.png"), 3)
    t4 = analyzeContours(findContoursInFile("../input/numbers/4.png"), 4)
    t5 = analyzeContours(findContoursInFile("../input/numbers/5.png"), 5)
    t6 = analyzeContours(findContoursInFile("../input/numbers/6.png"), 6)
    t7 = analyzeContours(findContoursInFile("../input/numbers/7.png"), 7)
    t8 = analyzeContours(findContoursInFile("../input/numbers/8.png"), 8)
    t9 = analyzeContours(findContoursInFile("../input/numbers/9.png"), 9)
    joined_list = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9

    # Set up training data
    labels = np.zeros(len(joined_list), dtype=int)
    trainingData = np.zeros((len(joined_list), len(joined_list[0]) - 3), dtype=np.float32)
    for idx, observation in enumerate(joined_list):
        labels[idx] = observation[0]
        trainingData[idx, 0] = observation[3]
        trainingData[idx, 1] = observation[4]
        trainingData[idx, 2] = observation[5]

    # Train the SVM
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_POLY)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    svm.setDegree(2)
    svm.setCoef0(2)
    svm.train(trainingData, cv2.ml.ROW_SAMPLE, labels)

    filename = "../input/sudokuer/01bw.png"
    image = cv2.imread(filename)
    output = analyzeContours(findContoursInFile(filename), 1)

    random.seed(12345)
    drawing = 0.0012*image
    for idx, values in enumerate(output):
        area = values[3]
        if area > 400:
            coordx = int(values[1])
            coordy = int(values[2])
            sampleMat = np.array([values[3:7]], dtype=np.float32)

            recognizedClass = int(svm.predict(sampleMat)[1][0][0])
            print(f"{coordx:5d}, {coordy:5d}, {recognizedClass:1}")
            cv2.putText(drawing, f"{recognizedClass}",
                    (coordx, coordy), 
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1, 
                    (0, 0, 255))

    cv2.namedWindow("Contours", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Contours", drawing)
    cv2.waitKey(-1)


main()
