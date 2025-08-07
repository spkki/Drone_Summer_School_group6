# Code from https://docs.opencv2.org/3.4/d1/d73/tutorial_introduction_to_svm.html
import cv2
import numpy as np
from icecream import ic

# Set up training data
labels = np.array([1, -1, -1, -1])
trainingData = np.matrix([[501, 10], [255, 10], [501, 255], [10, 501]], dtype=np.float32)

# Train the SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm.train(trainingData, cv2.ml.ROW_SAMPLE, labels)
ic(trainingData)
ic(labels)
ic(trainingData.shape)
ic(labels.shape)


# Data for visual representation
width = 512
height = 512
image = np.zeros((height, width, 3), dtype=np.uint8)

# Show the decision regions given by the SVM
green = (0, 255, 0)
blue = (255, 0, 0)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        sampleMat = np.matrix([[j,i]], dtype=np.float32)
        response = svm.predict(sampleMat)[1]
        if response == 1:
            image[i,j] = green
        elif response == -1:
            image[i,j] = blue

# Show the training data
thickness = -1
cv2.circle(image, (501,  10), 5, (  0,   0,   0), thickness)
cv2.circle(image, (255,  10), 5, (255, 255, 255), thickness)
cv2.circle(image, (501, 255), 5, (255, 255, 255), thickness)
cv2.circle(image, ( 10, 501), 5, (255, 255, 255), thickness)

# Show support vectors
thickness = 2
sv = svm.getUncompressedSupportVectors()
for i in range(sv.shape[0]):
    cv2.circle(image, (int(sv[i,0]), int(sv[i,1])), 6, (128, 128, 128), thickness)

cv2.imwrite('../output/ex05_result.png', image) # save the image
cv2.imshow('SVM Simple Example', image) # show it to the user
cv2.waitKey()
