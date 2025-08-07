import cv2

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
        observation.append(objectClass)
        observation.append(area)
        observation.append(perimeter)
        observation.append(mu['mu12'])
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
    print(joined_list)



main()
